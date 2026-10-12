"""Algoritmo genético para el problema del viajero (TSP)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .common import BaseOptimizer


class GeneticOptimizer(BaseOptimizer):
    """Algoritmo genético basado en permutaciones de ciudades."""

    name = "genetic"

    def _get_parameters(self) -> dict[str, Any]:
        """Obtiene y valida los parámetros configurables del algoritmo."""

        config = self.config or {}

        if not isinstance(config, Mapping):
            raise TypeError(
                "La configuración del algoritmo genético debe ser un diccionario."
            )

        # Permite recibir directamente los parámetros o una sección 'genetic'.
        if "genetic" in config:
            config = config["genetic"]

            if not isinstance(config, Mapping):
                raise TypeError(
                    "La sección 'genetic' debe ser un diccionario."
                )

        population_size = int(config.get("population_size", 60))
        crossover_rate = float(config.get("crossover_rate", 0.8))
        mutation_rate = float(config.get("mutation_rate", 0.15))
        tournament_size = int(config.get("tournament_size", 3))
        elite_count = int(config.get("elite_count", 2))

        if population_size < 2:
            raise ValueError("population_size debe ser >= 2.")

        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate debe estar entre 0 y 1.")

        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate debe estar entre 0 y 1.")

        if tournament_size < 2:
            raise ValueError("tournament_size debe ser >= 2.")

        if tournament_size > population_size:
            raise ValueError(
                "tournament_size no puede ser mayor que population_size."
            )

        if elite_count < 0 or elite_count >= population_size:
            raise ValueError(
                "elite_count debe ser >= 0 y menor que population_size."
            )

        return {
            "population_size": population_size,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "tournament_size": tournament_size,
            "elite_count": elite_count,
        }

    def _reset_state(self) -> None:
        """Reinicia el estado propio del algoritmo genético."""

        self.params = self._get_parameters()

    def _create_individual(self) -> np.ndarray:
        """Crea una ruta aleatoria válida."""

        return self.rng.permutation(self.n)

    def _create_population(self, size: int) -> list[np.ndarray]:
        """Crea una población de rutas válidas."""

        return [self._create_individual() for _ in range(size)]

    def _tournament_selection(
        self,
        population: list[np.ndarray],
        costs: np.ndarray,
    ) -> np.ndarray:
        """Selecciona un individuo mediante torneo."""

        tournament_size = min(
            self.params["tournament_size"],
            len(population),
        )

        candidates = self.rng.choice(
            len(population),
            size=tournament_size,
            replace=False,
        )

        winner_index = min(
            candidates,
            key=lambda index: costs[index],
        )

        return population[winner_index].copy()

    def _order_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cruza dos rutas usando Order Crossover (OX)."""

        # Según la tasa de cruce, puede que los padres pasen sin cruzarse.
        if self.rng.random() >= self.params["crossover_rate"]:
            return parent1.copy(), parent2.copy()

        # Elegimos dos puntos de corte distintos.
        cut1, cut2 = sorted(
            self.rng.choice(self.n, size=2, replace=False)
        )

        def build_child(
            first_parent: np.ndarray,
            second_parent: np.ndarray,
        ) -> np.ndarray:
            """Construye un hijo conservando un segmento del primer padre."""

            child = np.full(self.n, -1, dtype=np.int64)

            # Copiamos el segmento del primer padre.
            child[cut1:cut2 + 1] = first_parent[cut1:cut2 + 1]

            used = set(child[cut1:cut2 + 1].tolist())

            # Recorremos el segundo padre desde la posición posterior al corte,
            # dando la vuelta al final del arreglo.
            second_order = np.concatenate(
                (
                    second_parent[cut2 + 1:],
                    second_parent[:cut2 + 1],
                )
            )

            remaining_genes = [
                city
                for city in second_order
                if city not in used
            ]

            # Posiciones libres del hijo, también en orden circular.
            free_positions = list(range(cut2 + 1, self.n))
            free_positions += list(range(0, cut1))

            for position, city in zip(
                free_positions,
                remaining_genes,
            ):
                child[position] = city

            return child

        child1 = build_child(parent1, parent2)
        child2 = build_child(parent2, parent1)

        return child1, child2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Aplica mutación por inversión a una ruta."""

        mutated = individual.copy()

        if self.rng.random() >= self.params["mutation_rate"]:
            return mutated

        cut1, cut2 = sorted(
            self.rng.choice(self.n, size=2, replace=False)
        )

        mutated[cut1:cut2 + 1] = mutated[cut1:cut2 + 1][::-1]

        return mutated

    def _evaluate_population(
        self,
        population: list[np.ndarray],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Evalúa los individuos que permita el presupuesto."""

        evaluated_population: list[np.ndarray] = []
        costs: list[float] = []

        for individual in population:
            if self.remaining <= 0:
                break

            cost = self.evaluate(individual)

            evaluated_population.append(individual.copy())
            costs.append(cost)

        return (
            evaluated_population,
            np.asarray(costs, dtype=np.float64),
        )

    def _rank_population(
        self,
        population: list[np.ndarray],
        costs: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Ordena la población desde el menor hasta el mayor costo."""

        order = np.argsort(costs)

        ranked_population = [
            population[index].copy()
            for index in order
        ]

        ranked_costs = costs[order].copy()

        return ranked_population, ranked_costs

    def _select_elites(
        self,
        population: list[np.ndarray],
        costs: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Conserva los mejores individuos de la población."""

        ranked_population, ranked_costs = self._rank_population(
            population,
            costs,
        )

        elite_count = min(
            self.params["elite_count"],
            len(ranked_population),
        )

        elites = [
            individual.copy()
            for individual in ranked_population[:elite_count]
        ]

        elite_costs = ranked_costs[:elite_count].copy()

        return elites, elite_costs
    def _create_children(
        self,
        population: list[np.ndarray],
        costs: np.ndarray,
        amount: int,
    ) -> list[np.ndarray]:
        """Genera hijos mediante selección, cruce y mutación."""

        children: list[np.ndarray] = []

        while len(children) < amount:
            parent1 = self._tournament_selection(
                population,
                costs,
            )

            parent2 = self._tournament_selection(
                population,
                costs,
            )

            child1, child2 = self._order_crossover(
                parent1,
                parent2,
            )

            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            children.append(child1)

            if len(children) < amount:
                children.append(child2)

        return children

    def _run(self) -> None:
        """Ejecuta el ciclo principal del algoritmo genético."""

        population_size = self.params["population_size"]

        # ---------------------------------------------------------------
        # 1. Población inicial
        # ---------------------------------------------------------------
        population = self._create_population(population_size)

        population, costs = self._evaluate_population(
            population
        )

        # Si el presupuesto no alcanzó para evaluar toda la población
        # inicial, terminamos con las evaluaciones que sí se pudieron hacer.
        if len(population) < population_size:
            return

        self.end_iteration()

        # ---------------------------------------------------------------
        # 2. Evolución de generaciones
        # ---------------------------------------------------------------
        while self.remaining > 0:
            elites, elite_costs = self._select_elites(
                population,
                costs,
            )

            children_needed = population_size - len(elites)

            children = self._create_children(
                population,
                costs,
                children_needed,
            )

            evaluated_children, children_costs = (
                self._evaluate_population(children)
            )

            # Los élites mantienen el costo que ya conocíamos.
            population = elites + evaluated_children

            costs = np.concatenate(
                (
                    elite_costs,
                    children_costs,
                )
            )

            # Si el budget se acabó a mitad de una generación,
            # no la contamos como generación completa.
            if len(evaluated_children) < len(children):
                return

            self.end_iteration()