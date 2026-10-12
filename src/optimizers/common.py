"""Contrato común de optimizadores y evaluador de rutas.

Contenido:
    - validate_cost_matrix / validate_route / route_cost : utilidades puras.
    - RouteEvaluator   : único lugar donde se calcula el costo del ciclo y se
                         cuenta una evaluación.
    - HistoryEntry, OptimizationResult : formato de salida compartido (ACO, GA).
    - BaseOptimizer    : clase base (patrón "template method") para ACO y GA.

Contrato con el runner (guía, sección 7):
    result = Algoritmo.optimize(cost_matrix, seed, budget, config)
    -> {"best_route", "best_cost", "evaluations", "iterations", "history", ...}

Regla de conteo de evaluaciones
    +1 por cada ruta COMPLETA válida evaluada (incluye el arco de retorno).
    Las rutas inválidas o rechazadas por falta de presupuesto NO cuentan.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import numpy as np

__all__ = [
    "BudgetExhausted",
    "HistoryEntry",
    "OptimizationResult",
    "BaseOptimizer",
    "RouteEvaluator",
    "route_cost",
    "validate_cost_matrix",
    "validate_route",
]


class BudgetExhausted(Exception):
    """Se lanza al pedir una evaluación cuando el presupuesto ya se agotó."""


# --------------------------------------------------------------------------- #
# Utilidades puras
# --------------------------------------------------------------------------- #
def validate_integer(value: Any, name: str, minimum: int | None = None) -> int:
    """Valida un entero (sin aceptar bool ni decimales) y, opcionalmente, un mínimo.

    Tipo incorrecto -> TypeError. Valor fuera de rango -> ValueError.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} debe ser un entero.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} debe ser un entero >= {minimum}.")
    return int(value)


def validate_cost_matrix(cost_matrix: Any) -> np.ndarray:
    """Devuelve una COPIA float64 de solo lectura, o lanza ValueError.

    Exige matriz cuadrada n x n (n >= 2), finita y sin valores negativos.
    No asume simetría. La diagonal también debe ser finita ("definida").
    """
    matrix = np.array(cost_matrix, dtype=np.float64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"La matriz de costos debe ser cuadrada, recibió {matrix.shape}")
    if matrix.shape[0] < 2:
        raise ValueError("Se necesitan al menos dos ciudades (matriz mínima 2x2)")
    if not np.isfinite(matrix).all():
        raise ValueError("La matriz de costos contiene NaN o infinitos")
    if (matrix < 0).any():
        raise ValueError("La matriz de costos contiene valores negativos")
    matrix.setflags(write=False)
    return matrix


def validate_route(route: Sequence[int] | np.ndarray, n: int) -> np.ndarray:
    """Verifica que `route` sea una permutación de 0..n-1; devuelve int64."""
    route = np.asarray(route)
    if route.ndim != 1 or route.size != n:
        raise ValueError(f"La ruta debe tener {n} ciudades, recibió forma {route.shape}")
    if not np.issubdtype(route.dtype, np.integer):
        raise ValueError("La ruta debe contener enteros (ids de ciudad)")
    if route.min() < 0 or route.max() >= n:
        raise ValueError(f"Identificadores fuera del rango [0, {n - 1}]")
    if (np.bincount(route, minlength=n) != 1).any():
        raise ValueError("Todas las ciudades deben ser visitadas exactamente una vez (permutación)")
    return route.astype(np.int64, copy=False)


def route_cost(cost_matrix: np.ndarray, route: Sequence[int] | np.ndarray) -> float:
    """Costo del ciclo cerrado (incluye retorno). NO cuenta evaluaciones.

    Úsala en tests para recalcular el costo reportado de forma independiente.
    """
    matrix = validate_cost_matrix(cost_matrix)
    r = validate_route(route, matrix.shape[0])
    return float(matrix[r, np.roll(r, -1)].sum())


def _snapshot(config: Any) -> dict[str, Any]:
    """Copia independiente de la configuración, para registrar el experimento."""
    if config is None:
        return {}
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)  # asdict ya copia en profundidad
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))
    raise TypeError("config debe ser una dataclass, un Mapping o None")


# --------------------------------------------------------------------------- #
# Evaluador común
# --------------------------------------------------------------------------- #
class RouteEvaluator:
    """Evalúa rutas, valida permutaciones y lleva el contador de evaluaciones."""

    def __init__(self, cost_matrix: Any, budget: int, *, validate: bool = True) -> None:
        self._cost_matrix = validate_cost_matrix(cost_matrix) if validate else cost_matrix
        self._n = int(self._cost_matrix.shape[0])
        self._budget = validate_integer(budget, "budget", minimum=1)
        self._evaluations = 0

    @property
    def n(self) -> int:
        return self._n

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def evaluations(self) -> int:
        return self._evaluations

    @property
    def remaining(self) -> int:
        return self._budget - self._evaluations

    def evaluate(self, route: Sequence[int] | np.ndarray) -> float:
        """Costo del ciclo. Lanza BudgetExhausted sin consumir si no hay saldo."""
        if self._evaluations >= self._budget:
            raise BudgetExhausted(f"Presupuesto de {self._budget} evaluaciones agotado")
        r = validate_route(route, self._n)  # si falla, no se cuenta
        cost = float(self._cost_matrix[r, np.roll(r, -1)].sum())
        self._evaluations += 1  # ÚNICO punto donde se incrementa el contador
        return cost


# --------------------------------------------------------------------------- #
# Formato de salida compartido
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HistoryEntry:
    """Mejor global conocido (best-so-far) tras un paso del algoritmo."""

    step: int
    evaluations: int
    best_cost: float
    best_route: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "evaluations": self.evaluations,
            "best_cost": self.best_cost,
            "best_route": list(self.best_route),
        }


@dataclass(frozen=True)
class OptimizationResult:
    best_route: tuple[int, ...]
    best_cost: float
    evaluations: int
    iterations: int
    history: tuple[HistoryEntry, ...]
    stop_reason: str  # "budget" | "completed"
    seed: int
    budget: int
    config: dict[str, Any]  # snapshot para reproducir la corrida

    def to_dict(self) -> dict[str, Any]:
        """Diccionario serializable a JSON, con las claves del contrato."""
        return {
            "best_route": list(self.best_route),
            "best_cost": self.best_cost,
            "evaluations": self.evaluations,
            "iterations": self.iterations,
            "history": [h.to_dict() for h in self.history],
            "stop_reason": self.stop_reason,
            "seed": self.seed,
            "budget": self.budget,
            "config": dict(self.config),
        }


# --------------------------------------------------------------------------- #
# Clase base
# --------------------------------------------------------------------------- #
class BaseOptimizer(ABC):
    """Esqueleto común. Cada algoritmo implementa solo `_run` (y `_reset_state`).

    Garantías que da la base (y que los tests del equipo pueden asumir):
      * única fuente de aleatoriedad: `self.rng`, creada desde `seed`;
      * `run()` siempre parte de cero (llamarlo dos veces da lo mismo);
      * las evaluaciones pasan por `self.evaluate`, nunca exceden el budget;
      * el historial es best-so-far (el costo nunca aumenta);
      * la mejor ruta es siempre una permutación válida.

    Uso desde una subclase:
        self.rng          generador numpy ya sembrado
        self.n            número de ciudades
        self.cost_matrix  matriz validada (solo lectura)
        self.config       parámetros específicos del algoritmo
        self.remaining    evaluaciones disponibles
        self.evaluate(r)  costo de la ruta r (actualiza el mejor global)
        self.end_iteration()  cierra un paso y lo registra en el historial
    """

    name = "base"

    def __init__(
        self,
        cost_matrix: Any,
        seed: int,
        budget: int,
        config: Any = None,
    ) -> None:
        self._cost_matrix = validate_cost_matrix(cost_matrix)
        self._n = int(self._cost_matrix.shape[0])
        self._seed = validate_integer(seed, "seed", minimum=0)
        self._budget = validate_integer(budget, "budget", minimum=1)
        self._config = config
        self._config_snapshot = _snapshot(config)

        # Estado común SIN invocar hooks: la subclase aún podría no estar
        # construida. El hook _reset_state() corre en reset(), al inicio de run().
        self._initialize_common_state()

    # ---- solo lectura para las subclases ------------------------------- #
    @property
    def n(self) -> int:
        return self._n

    @property
    def cost_matrix(self) -> np.ndarray:
        return self._cost_matrix

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def config(self) -> Any:
        return self._config

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @property
    def evaluations(self) -> int:
        return self._evaluator.evaluations

    @property
    def remaining(self) -> int:
        return self._evaluator.remaining

    @property
    def best_route(self) -> tuple[int, ...] | None:
        return self._best_route

    @property
    def best_cost(self) -> float:
        return self._best_cost

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def history(self) -> tuple[HistoryEntry, ...]:
        return tuple(self._history)

    # ---- ciclo de vida -------------------------------------------------- #
    def _initialize_common_state(self) -> None:
        """Inicializa el estado compartido (rng, evaluador, mejor, historial)."""
        self._rng = np.random.default_rng(self._seed)
        self._evaluator = RouteEvaluator(self._cost_matrix, self._budget, validate=False)
        self._best_route: tuple[int, ...] | None = None
        self._best_cost: float = float("inf")
        self._history: list[HistoryEntry] = []
        self._iterations: int = 0

    def _reset_state(self) -> None:
        """Hook: la subclase inicializa su estado (p. ej. la matriz τ de ACO).

        Se ejecuta al inicio de cada run(), no durante la construcción.
        """

    def reset(self) -> None:
        """Reinicia el estado mutable; recrea el generador desde la semilla."""
        self._initialize_common_state()
        self._reset_state()

    @abstractmethod
    def _run(self) -> None:
        """Bucle principal del algoritmo.

        Debe usar `self.evaluate` para evaluar y `self.end_iteration` al cerrar
        cada paso. Puede consultar `self.remaining` para decidir qué hacer con
        iteraciones parciales. Si se agota el budget, `evaluate` lanza
        BudgetExhausted y `run` lo captura.
        """

    def run(self) -> OptimizationResult:
        """Ejecuta el algoritmo y devuelve el resultado estandarizado."""
        self.reset()
        stop_reason = "completed"
        try:
            self._run()
        except BudgetExhausted:
            stop_reason = "budget"
        else:
            if self._evaluator.remaining == 0:
                stop_reason = "budget"

        # Evaluaciones hechas después del último cierre (iteración parcial):
        # se registran en el historial pero NO suman una iteración.
        last = self._history[-1].evaluations if self._history else 0
        if self._evaluator.evaluations > last:
            self._record_history()

        if self._best_route is None:
            raise RuntimeError("El optimizador terminó sin evaluar ninguna ruta")

        return OptimizationResult(
            best_route=self._best_route,
            best_cost=self._best_cost,
            evaluations=self._evaluator.evaluations,
            iterations=self._iterations,
            history=tuple(self._history),
            stop_reason=stop_reason,
            seed=self._seed,
            budget=self._budget,
            config=copy.deepcopy(self._config_snapshot),
        )

    # ---- servicios para las subclases ----------------------------------- #
    def evaluate(self, route: Sequence[int] | np.ndarray) -> float:
        """Evalúa una ruta; actualiza el mejor global solo si mejora ESTRICTAMENTE."""
        cost = self._evaluator.evaluate(route)  # valida, cuenta y respeta budget
        if cost < self._best_cost:
            self._best_cost = cost
            self._best_route = tuple(np.asarray(route).tolist())
        return cost

    def _record_history(self) -> None:
        """Registra el estado actual en el historial (best-so-far)."""
        self._history.append(
            HistoryEntry(
                step=len(self._history),
                evaluations=self._evaluator.evaluations,
                best_cost=self._best_cost,
                best_route=self._best_route,
            )
        )

    def end_iteration(self) -> None:
        """Cierra un paso (iteración ACO, generación GA) y lo anota en el historial."""
        if self._best_route is None:
            raise RuntimeError("end_iteration() llamado sin haber evaluado rutas")
        self._record_history()
        self._iterations += 1

    # ---- punto de entrada del contrato ---------------------------------- #
    @classmethod
    def optimize(
        cls,
        cost_matrix: Any,
        seed: int,
        budget: int,
        config: Any = None,
    ) -> dict[str, Any]:
        """optimize(cost_matrix, seed, budget, config) -> dict del contrato."""
        return cls(cost_matrix, seed, budget, config).run().to_dict()