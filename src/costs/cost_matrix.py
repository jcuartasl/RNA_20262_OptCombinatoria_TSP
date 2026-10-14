"""Matriz de costos por arco dirigido (Persona 4).

    C_ij(v) = v * T_ij + P_ij + F_ij
        v     valor de la hora del vendedor  [EUR/h]
        T_ij  tiempo de viaje                [h]
        P_ij  peajes                         [EUR]
        F_ij  combustible                    [EUR]  (ver vehicle.py)

La matriz base (T, D, P, F) es independiente de v. Los peajes faltantes NO se interpretan
como 0: con policy="error" (por defecto) se lanza PendingDataError.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.road_matrix import PROCESSED_DIR, load_cities, load_distance_matrix, load_time_matrix
from ..optimizers.common import validate_route
from .vehicle import PendingDataError, Vehicle, fuel_cost_matrix

__all__ = [
    "CostComponents",
    "RouteTotals",
    "TOLL_FILENAME",
    "build_components",
    "load_components",
    "load_toll_matrix",
]

TOLL_FILENAME = "toll_eur.csv"
_TOLL_POLICIES = ("error", "provisional_zero")


def _check_hourly_value(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError("hourly_value_eur debe ser numérico.")
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"hourly_value_eur debe ser finito y >= 0 (recibió {value!r}).")
    return v


@dataclass(frozen=True)
class RouteTotals:
    """Totales de un ciclo cerrado, por componente. Costo(v) = v*time_h + fixed_eur."""

    time_h: float
    distance_km: float
    toll_eur: float
    fuel_eur: float

    @property
    def fixed_eur(self) -> float:
        """K = P + F: parte del costo que no depende de v."""
        return self.toll_eur + self.fuel_eur

    def cost(self, hourly_value_eur: float) -> float:
        return _check_hourly_value(hourly_value_eur) * self.time_h + self.fixed_eur


@dataclass(frozen=True, eq=False)
class CostComponents:
    city_names: tuple[str, ...]
    time_h: np.ndarray
    distance_km: np.ndarray
    toll_eur: np.ndarray
    fuel_eur: np.ndarray
    vehicle: Vehicle
    tolls_complete: bool = True

    @property
    def n(self) -> int:
        return len(self.city_names)

    @property
    def fixed_eur(self) -> np.ndarray:
        return self.toll_eur + self.fuel_eur

    def cost_matrix(self, hourly_value_eur: float) -> np.ndarray:
        """C(v) = v*T + P + F (nueva matriz n x n; diagonal 0)."""
        v = _check_hourly_value(hourly_value_eur)
        return v * self.time_h + self.toll_eur + self.fuel_eur

    def route_totals(self, route: Sequence[int] | np.ndarray) -> RouteTotals:
        """Totales por componente del ciclo cerrado (incluye el arco de retorno)."""
        r = validate_route(route, self.n)
        src, dst = r, np.roll(r, -1)
        return RouteTotals(
            time_h=float(self.time_h[src, dst].sum()),
            distance_km=float(self.distance_km[src, dst].sum()),
            toll_eur=float(self.toll_eur[src, dst].sum()),
            fuel_eur=float(self.fuel_eur[src, dst].sum()),
        )


def _as_square(name: str, values: Any) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        raise ValueError(f"{name} debe ser una matriz cuadrada (n >= 2), recibió {arr.shape}")
    return arr


def build_components(
    time_h: Any,
    distance_km: Any,
    toll_eur: Any,
    vehicle: Vehicle,
    city_names: Sequence[str] | None = None,
    toll_policy: str = "error",
) -> CostComponents:
    """Valida unidades/forma y arma los componentes. Peajes NaN = dato PENDIENTE."""
    if toll_policy not in _TOLL_POLICIES:
        raise ValueError(f"toll_policy debe ser uno de {_TOLL_POLICIES}")

    T = _as_square("time_h", time_h)
    D = _as_square("distance_km", distance_km)
    P = _as_square("toll_eur", toll_eur)
    if not (T.shape == D.shape == P.shape):
        raise ValueError(f"Formas distintas: T{T.shape}, D{D.shape}, P{P.shape}")
    n = T.shape[0]

    names = tuple(city_names) if city_names is not None else tuple(str(i) for i in range(n))
    if len(names) != n:
        raise ValueError(f"Se esperaban {n} nombres de ciudad, recibió {len(names)}")

    off = ~np.eye(n, dtype=bool)
    for label, M in (("time_h", T), ("distance_km", D)):
        if not np.isfinite(M[off]).all():
            raise ValueError(f"{label} contiene NaN o infinitos fuera de la diagonal")
        if (M[off] < 0).any():
            raise ValueError(f"{label} contiene valores negativos")
        M[np.diag_indices(n)] = 0.0

    if np.isinf(P).any():
        raise ValueError("toll_eur contiene infinitos")
    missing = np.isnan(P) & off
    n_missing = int(missing.sum())
    tolls_complete = True
    if n_missing:
        if toll_policy == "error":
            raise PendingDataError(
                f"Faltan {n_missing} de {n * (n - 1)} peajes (PENDIENTES). "
                "No se interpretan como 0."
            )
        warnings.warn(
            f"{n_missing} peajes faltantes se rellenan con 0 de forma PROVISIONAL: "
            "no usar para resultados finales.",
            UserWarning,
            stacklevel=2,
        )
        P[missing] = 0.0
        tolls_complete = False
    P[np.diag_indices(n)] = 0.0
    if (P < 0).any():
        raise ValueError("toll_eur contiene valores negativos")

    F = fuel_cost_matrix(D, vehicle)
    for M in (T, D, P, F):
        M.setflags(write=False)
    return CostComponents(names, T, D, P, F, vehicle, tolls_complete)


def load_toll_matrix(
    data_dir: str | Path = PROCESSED_DIR,
    city_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Lee toll_eur.csv (n x n, mismo orden canónico). Celdas vacías -> NaN (pendiente)."""
    path = Path(data_dir) / TOLL_FILENAME
    if not path.exists():
        raise PendingDataError(f"No existe {path}: peajes PENDIENTES.")
    try:
        df = pd.read_csv(path, index_col=0)
    except pd.errors.EmptyDataError as exc:
        raise PendingDataError(f"{path} está vacío: peajes PENDIENTES.") from exc

    if city_names is None:
        city_names = load_cities(str(data_dir))["name"].tolist()
    names = list(city_names)
    if df.shape != (len(names), len(names)):
        raise ValueError(f"toll_eur.csv debe ser {len(names)}x{len(names)}, es {df.shape}")
    if list(df.index) != names or list(df.columns) != names:
        raise ValueError("Filas/columnas de toll_eur.csv no coinciden con el orden canónico")
    return df.to_numpy(dtype=np.float64)


def load_components(
    vehicle: Vehicle,
    data_dir: str | Path = PROCESSED_DIR,
    toll_policy: str = "error",
) -> CostComponents:
    """Carga T, D y P desde disco (offline) y construye los componentes."""
    names = load_cities(str(data_dir))["name"].tolist()
    return build_components(
        load_time_matrix(str(data_dir)),
        load_distance_matrix(str(data_dir)),
        load_toll_matrix(data_dir, names),
        vehicle,
        city_names=names,
        toll_policy=toll_policy,
    )
