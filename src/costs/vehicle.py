"""Vehículo y costo de combustible por arco (Persona 4).

Modelo:   F_ij = (D_ij / 100) * q * p
    D_ij  distancia por carretera del arco i->j      [km]
    q     consumo HOMOLOGADO del vehículo            [L/100 km]
    p     precio del combustible                     [EUR/L]
    F_ij  costo de combustible del arco              [EUR]

Aquí no se inventa ningún dato: mientras el vehículo, el consumo, el precio, sus fuentes o
la fecha estén PENDIENTE, `Vehicle` / `load_vehicle` lanzan PendingDataError.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

__all__ = ["PendingDataError", "Vehicle", "fuel_cost_matrix", "is_pending", "load_vehicle"]

_REQUIRED_TEXT = (
    "model",
    "fuel_type",
    "consumption_source",
    "fuel_price_source",
    "fuel_price_date",
)
_REQUIRED_NUMBERS = ("consumption_l_per_100km", "fuel_price_eur_per_l")


class PendingDataError(ValueError):
    """Un dato requerido sigue PENDIENTE (sin valor o sin fuente verificable)."""


def is_pending(value: Any) -> bool:
    """True si el valor es None, vacío o un marcador PENDIENTE*."""
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        return text == "" or text.upper().startswith("PENDIENTE")
    return False


def _check_positive(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} debe ser numérico.")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} debe ser un número finito > 0 (recibió {value!r}).")


@dataclass(frozen=True)
class Vehicle:
    model: str
    fuel_type: str
    consumption_l_per_100km: float
    fuel_price_eur_per_l: float
    consumption_source: str
    fuel_price_source: str
    fuel_price_date: str
    year: int | None = None

    def __post_init__(self) -> None:
        pending = [n for n in _REQUIRED_TEXT + _REQUIRED_NUMBERS if is_pending(getattr(self, n))]
        if pending:
            raise PendingDataError(
                "Datos del vehículo PENDIENTES: " + ", ".join(pending)
                + ". Complétalos con fuente y fecha en config/sensitivity.yaml."
            )
        for name in _REQUIRED_NUMBERS:
            _check_positive(name, getattr(self, name))

    @property
    def fuel_cost_per_km(self) -> float:
        """EUR por km recorrido: (q / 100) * p."""
        return self.consumption_l_per_100km / 100.0 * self.fuel_price_eur_per_l


def fuel_cost_matrix(distance_km: Any, vehicle: Vehicle) -> np.ndarray:
    """Matriz F_ij = (D_ij / 100) * q * p, en EUR."""
    dist = np.array(distance_km, dtype=np.float64)
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"distance_km debe ser cuadrada, recibió {dist.shape}")
    return dist / 100.0 * vehicle.consumption_l_per_100km * vehicle.fuel_price_eur_per_l


def load_vehicle(source: Mapping[str, Any] | str | os.PathLike[str]) -> Vehicle:
    """Construye el Vehicle desde un dict (con o sin clave 'vehicle') o una ruta a YAML."""
    if isinstance(source, (str, os.PathLike)):
        import yaml

        with open(source, "r", encoding="utf-8") as handle:
            source = yaml.safe_load(handle)
    if not isinstance(source, Mapping):
        raise TypeError("La configuración del vehículo debe ser un diccionario o un YAML.")
    section = source.get("vehicle", source)
    if not isinstance(section, Mapping):
        raise TypeError("La sección 'vehicle' debe ser un diccionario.")
    allowed = {f.name for f in fields(Vehicle)}
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"Parámetros de vehículo desconocidos: {unknown}")
    return Vehicle(**{name: section.get(name) for name in allowed})
