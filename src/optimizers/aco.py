"""Colonia de hormigas (ACO, Ant System) para el problema del viajero (TSP).

Ecuaciones
----------
Visibilidad:   eta_ij = 1 / max(C_ij, eps)               (eta_ii = 0)
Transición:    p_ij ∝ tau_ij^alpha * eta_ij^beta         (solo ciudades no visitadas)
Evaporación:   tau_ij <- (1 - rho) * tau_ij
Depósito:      tau_ij <- tau_ij + sum_k q / L_k          por cada arco i->j de la ruta k
               (el arco de cierre, última -> primera, también deposita)
Inicial:       tau_0 = n_ants / L_nn                     (L_nn: vecino más cercano)
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .common import BaseOptimizer, route_cost

# Todos los parámetros son obligatorios: los valores salen del YAML (aco.yaml)
# y el código no define ninguno por defecto.
REQUIRED_PARAMETERS = ("n_ants", "alpha", "beta", "rho", "q", "eps")


def _to_float(name: str, value: Any) -> float:
    """Convierte a flotante finito (PyYAML entrega '1e-9' como texto)."""

    if isinstance(value, bool) or value is None:
        raise TypeError(f"{name} debe ser numérico, recibió {value!r}.")

    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} debe ser numérico, recibió {value!r}.") from None

    if not math.isfinite(number):
        raise ValueError(f"{name} debe ser finito, recibió {value!r}.")

    return number


def _to_int(name: str, value: Any) -> int:
    """Convierte a entero, rechazando valores con parte decimal."""

    number = _to_float(name, value)

    if not number.is_integer():
        raise ValueError(f"{name} debe ser un entero, recibió {value!r}.")

    return int(number)


class ACO(BaseOptimizer):
    """Colonia de hormigas basada en construcción de rutas (Ant System)."""

    name = "aco"

    def __init__(
        self,
        cost_matrix: Any,
        seed: int,
        budget: int,
        config: Any = None,
    ) -> None:
        if isinstance(config, (str, os.PathLike)):
            config = self._load_yaml(config)

        super().__init__(cost_matrix, seed, budget, config)

        self.params = self._get_parameters()

    def _get_parameters(self) -> dict[str, Any]:
        """Obtiene y valida los parámetros configurables del algoritmo."""

        config = self.config or {}

        if config is None:
            raise ValueError(
                "ACO requiere una configuración (diccionario o ruta a un YAML). "
                "El código no define valores por defecto."
            )

        if isinstance(config, (str, os.PathLike)):
            config = self._load_yaml(config)

        if not isinstance(config, Mapping):
            raise TypeError("La configuración de ACO debe ser un diccionario.")

        # Permite recibir directamente los parámetros o una sección 'aco'.
        if "aco" in config:
            config = config["aco"]

            if not isinstance(config, Mapping):
                raise TypeError("La sección 'aco' debe ser un diccionario.")

        # Las claves que empiezan con "_" se tratan como comentarios.
        config = {
            key: value
            for key, value in config.items()
            if not str(key).startswith("_")
        }

        unknown = sorted(set(config) - set(REQUIRED_PARAMETERS))
        if unknown:
            raise ValueError(f"Parámetros desconocidos para ACO: {unknown}.")

        missing = [key for key in REQUIRED_PARAMETERS if key not in config]
        if missing:
            raise ValueError(f"Faltan parámetros de ACO: {missing}.")

        n_ants = _to_int("n_ants", config["n_ants"])
        alpha = _to_float("alpha", config["alpha"])
        beta = _to_float("beta", config["beta"])
        rho = _to_float("rho", config["rho"])
        q = _to_float("q", config["q"])
        eps = _to_float("eps", config["eps"])

        if n_ants < 1:
            raise ValueError("n_ants debe ser >= 1.")

        if alpha < 0.0 or beta < 0.0:
            raise ValueError("alpha y beta deben ser >= 0.")

        if not 0.0 < rho <= 1.0:
            raise ValueError("rho debe estar en (0, 1].")

        if q <= 0.0:
            raise ValueError("q debe ser > 0.")

        if eps <= 0.0:
            raise ValueError("eps debe ser > 0.")

        return {
            "n_ants": n_ants,
            "alpha": alpha,
            "beta": beta,
            "rho": rho,
            "q": q,
            "eps": eps,
        }

    @staticmethod
    def _load_yaml(path: str | os.PathLike[str]) -> Mapping[str, Any]:
        """Lee un archivo YAML de parámetros."""

        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Leer el YAML requiere PyYAML: pip install pyyaml") from exc

        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        # Un archivo vacío se informa como "faltan parámetros".
        return {} if data is None else data

    def _reset_state(self) -> None:
        """Reinicia el estado propio de la colonia de hormigas."""

        self.params = self._get_parameters()

        self.eta = self._build_heuristic()

        initial_tau = self.params["n_ants"] / self._nearest_neighbor_cost()
        self.tau = np.full((self.n, self.n), initial_tau, dtype=np.float64)

    def _build_heuristic(self) -> np.ndarray:
        """Calcula eta^beta con eta = 1 / max(costo, eps), normalizada a [0, 1].

        La diagonal vale 0 (no es un traslado válido) y los arcos con costo
        infinito quedan con visibilidad 0. Normalizar por el máximo fuera de la
        diagonal evita desbordamientos con beta alto y hace que las
        probabilidades no dependan de la escala de los costos.
        """

        with np.errstate(divide="ignore"):
            eta = 1.0 / np.maximum(self.cost_matrix, self.params["eps"])

        np.fill_diagonal(eta, 0.0)

        off_diagonal = ~np.eye(self.n, dtype=bool)
        valid = eta[off_diagonal & np.isfinite(eta)]

        if valid.size > 0 and valid.max() > 0.0:
            eta = eta / valid.max()

        return eta ** self.params["beta"]

    def _nearest_neighbor_cost(self) -> float:
        """Costo de la ruta del vecino más cercano.

        Solo fija la escala inicial de la feromona: no pasa por `evaluate`, por
        lo que no consume presupuesto ni compite como solución. Si el costo es
        inválido (0 o infinito por arcos prohibidos) se usa n veces el costo
        medio de los arcos finitos y, en último caso, 1.0.
        """

        matrix = self.cost_matrix
        visited = np.zeros(self.n, dtype=bool)
        route = np.empty(self.n, dtype=np.int64)

        current = 0
        route[0] = current
        visited[current] = True

        for step in range(1, self.n):
            current = int(np.argmin(np.where(visited, np.inf, matrix[current])))
            route[step] = current
            visited[current] = True

        cost = float(route_cost(matrix, route))

        if math.isfinite(cost) and cost > 0.0:
            return cost

        finite = matrix[~np.eye(self.n, dtype=bool) & np.isfinite(matrix)]
        fallback = float(finite.mean()) * self.n if finite.size else 0.0

        return fallback if math.isfinite(fallback) and fallback > 0.0 else 1.0

    def _construct_routes(self, ants: int) -> np.ndarray:
        """Construye una ruta por hormiga, todas en paralelo (vectorizado)."""

        n, rng = self.n, self.rng

        # tau se normaliza por su máximo: las probabilidades no cambian y se
        # evita el desbordamiento con alpha alto.
        attraction = (
            (self.tau / self.tau.max()) ** self.params["alpha"] * self.eta
        )

        routes = np.empty((ants, n), dtype=np.int64)
        rows = np.arange(ants)

        current = rng.integers(0, n, size=ants)
        routes[:, 0] = current

        unvisited = np.ones((ants, n), dtype=bool)
        unvisited[rows, current] = False

        for step in range(1, n):
            # Las ciudades visitadas tienen peso exactamente 0 por máscara.
            weights = attraction[current] * unvisited
            totals = weights.sum(axis=1)

            # Si todos los pesos se anularon (underflow, arcos prohibidos),
            # se reparte uniformemente entre las ciudades no visitadas.
            stuck = ~(totals > 0.0)
            if stuck.any():
                weights[stuck] = unvisited[stuck]
                totals[stuck] = weights[stuck].sum(axis=1)

            # Ruleta: primer índice cuya suma acumulada supera el sorteo. El
            # tope evita elegir una ciudad sin peso por redondeo.
            cumulative = np.cumsum(weights, axis=1)
            draw = rng.random(ants) * totals
            last_valid = n - 1 - np.argmax(weights[:, ::-1] > 0.0, axis=1)

            chosen = np.minimum(
                (cumulative <= draw[:, None]).sum(axis=1),
                last_valid,
            )

            routes[:, step] = chosen
            unvisited[rows, chosen] = False
            current = chosen

        return routes

    def _evaluate_routes(
        self,
        routes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evalúa las rutas que permita el presupuesto."""

        evaluated_routes: list[np.ndarray] = []
        costs: list[float] = []

        for route in routes:
            if self.remaining <= 0:
                break

            cost = self.evaluate(route)

            evaluated_routes.append(route.copy())
            costs.append(cost)

        return (
            np.asarray(evaluated_routes, dtype=np.int64).reshape(-1, self.n),
            np.asarray(costs, dtype=np.float64),
        )

    def _evaporate(self) -> None:
        """Evapora la feromona: tau <- (1 - rho) * tau."""

        self.tau *= 1.0 - self.params["rho"]

    def _deposit(self, routes: np.ndarray, costs: np.ndarray) -> None:
        """Cada hormiga suma q / costo en los arcos de su ruta (incluido el de cierre)."""

        amounts = self.params["q"] / np.maximum(costs, self.params["eps"])

        source = routes
        target = np.roll(routes, -1, axis=1)  # el último regresa a la primera

        delta = np.bincount(
            (source * self.n + target).ravel(),
            weights=np.repeat(amounts, self.n),
            minlength=self.n * self.n,
        ).reshape(self.n, self.n)

        self.tau += delta

    def _update_pheromone(self, routes: np.ndarray, costs: np.ndarray) -> None:
        """Evapora y luego deposita la feromona de todas las hormigas."""

        self._evaporate()
        self._deposit(routes, costs)

    def _run(self) -> None:
        """Ejecuta el ciclo principal de la colonia de hormigas."""

        n_ants = self.params["n_ants"]

        while self.remaining > 0:
            # -----------------------------------------------------------
            # 1. Construcción de rutas (solo las que se pueden evaluar)
            # -----------------------------------------------------------
            ants = min(n_ants, self.remaining)
            routes = self._construct_routes(ants)

            # -----------------------------------------------------------
            # 2. Evaluación
            # -----------------------------------------------------------
            routes, costs = self._evaluate_routes(routes)

            # Si el budget se acabó a mitad de una iteración,
            # no la contamos como iteración completa.
            if len(routes) < n_ants:
                return

            # -----------------------------------------------------------
            # 3. Actualización de feromonas
            # -----------------------------------------------------------
            self._update_pheromone(routes, costs)

            self.end_iteration()