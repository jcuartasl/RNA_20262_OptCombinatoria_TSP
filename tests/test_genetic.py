"""Pruebas del algoritmo genético para el TSP."""

import numpy as np
import pytest

from src.optimizers.common import route_cost
from src.optimizers.genetic import GeneticOptimizer


def toy_matrix() -> np.ndarray:
    """Matriz pequeña y controlada para las pruebas."""

    return np.array(
        [
            [0, 1, 5, 9, 9],
            [1, 0, 1, 5, 9],
            [5, 1, 0, 1, 5],
            [9, 5, 1, 0, 1],
            [1, 9, 5, 1, 0],
        ],
        dtype=float,
    )


def asymmetric_matrix() -> np.ndarray:
    """Matriz dirigida para comprobar que no se asume simetría."""

    return np.array(
        [
            [0, 1, 8, 4],
            [7, 0, 2, 9],
            [3, 6, 0, 1],
            [5, 2, 7, 0],
        ],
        dtype=float,
    )


def base_config() -> dict:
    """Configuración pequeña para pruebas rápidas."""

    return {
        "population_size": 4,
        "crossover_rate": 0.8,
        "mutation_rate": 0.15,
        "tournament_size": 3,
        "elite_count": 2,
    }


def test_population_contains_valid_permutations():
    ga = GeneticOptimizer(
        cost_matrix=toy_matrix(),
        seed=42,
        budget=100,
        config=base_config(),
    )

    ga.reset()

    population = ga._create_population(10)

    for individual in population:
        assert sorted(individual.tolist()) == [0, 1, 2, 3, 4]


def test_order_crossover_preserves_permutation():
    config = base_config()
    config["crossover_rate"] = 1.0

    ga = GeneticOptimizer(
        cost_matrix=toy_matrix(),
        seed=42,
        budget=100,
        config=config,
    )

    ga.reset()

    parent1 = np.array([0, 1, 2, 3, 4])
    parent2 = np.array([4, 2, 0, 1, 3])

    child1, child2 = ga._order_crossover(
        parent1,
        parent2,
    )

    assert sorted(child1.tolist()) == [0, 1, 2, 3, 4]
    assert sorted(child2.tolist()) == [0, 1, 2, 3, 4]


def test_mutation_preserves_permutation():
    config = base_config()
    config["mutation_rate"] = 1.0

    ga = GeneticOptimizer(
        cost_matrix=toy_matrix(),
        seed=42,
        budget=100,
        config=config,
    )

    ga.reset()

    individual = np.array([0, 1, 2, 3, 4])

    mutated = ga._mutate(individual)

    assert sorted(mutated.tolist()) == [0, 1, 2, 3, 4]


def test_algorithm_respects_budget():
    result = GeneticOptimizer.optimize(
        cost_matrix=toy_matrix(),
        seed=42,
        budget=20,
        config=base_config(),
    )

    assert result["evaluations"] <= 20
    assert result["evaluations"] == 20
    assert result["stop_reason"] == "budget"


def test_best_route_is_valid():
    result = GeneticOptimizer.optimize(
        cost_matrix=toy_matrix(),
        seed=42,
        budget=20,
        config=base_config(),
    )

    assert sorted(result["best_route"]) == [0, 1, 2, 3, 4]


def test_reported_cost_matches_route_cost():
    matrix = toy_matrix()

    result = GeneticOptimizer.optimize(
        cost_matrix=matrix,
        seed=42,
        budget=20,
        config=base_config(),
    )

    recalculated_cost = route_cost(
        matrix,
        result["best_route"],
    )

    assert np.isclose(
        result["best_cost"],
        recalculated_cost,
    )


def test_history_is_best_so_far():
    result = GeneticOptimizer.optimize(
        cost_matrix=toy_matrix(),
        seed=42,
        budget=30,
        config=base_config(),
    )

    costs = [
        entry["best_cost"]
        for entry in result["history"]
    ]

    assert all(
        costs[i] <= costs[i - 1]
        for i in range(1, len(costs))
    )


def test_same_seed_is_reproducible():
    result1 = GeneticOptimizer.optimize(
        cost_matrix=toy_matrix(),
        seed=123,
        budget=30,
        config=base_config(),
    )

    result2 = GeneticOptimizer.optimize(
        cost_matrix=toy_matrix(),
        seed=123,
        budget=30,
        config=base_config(),
    )

    assert result1["best_route"] == result2["best_route"]
    assert result1["best_cost"] == result2["best_cost"]
    assert result1["evaluations"] == result2["evaluations"]
    assert result1["history"] == result2["history"]


def test_small_budget_is_respected():
    result = GeneticOptimizer.optimize(
        cost_matrix=toy_matrix(),
        seed=42,
        budget=2,
        config=base_config(),
    )

    assert result["evaluations"] == 2
    assert result["evaluations"] <= result["budget"]


def test_asymmetric_matrix():
    matrix = asymmetric_matrix()

    config = {
        "population_size": 4,
        "crossover_rate": 0.8,
        "mutation_rate": 0.15,
        "tournament_size": 2,
        "elite_count": 1,
    }

    result = GeneticOptimizer.optimize(
        cost_matrix=matrix,
        seed=7,
        budget=20,
        config=config,
    )

    assert sorted(result["best_route"]) == [0, 1, 2, 3]

    assert np.isclose(
        result["best_cost"],
        route_cost(matrix, result["best_route"]),
    )


def test_invalid_population_size():
    config = base_config()
    config["population_size"] = 1

    with pytest.raises(ValueError):
        GeneticOptimizer.optimize(
            cost_matrix=toy_matrix(),
            seed=42,
            budget=20,
            config=config,
        )


def test_invalid_crossover_rate():
    config = base_config()
    config["crossover_rate"] = 1.5

    with pytest.raises(ValueError):
        GeneticOptimizer.optimize(
            cost_matrix=toy_matrix(),
            seed=42,
            budget=20,
            config=config,
        )


def test_invalid_mutation_rate():
    config = base_config()
    config["mutation_rate"] = -0.1

    with pytest.raises(ValueError):
        GeneticOptimizer.optimize(
            cost_matrix=toy_matrix(),
            seed=42,
            budget=20,
            config=config,
        )


def test_tournament_cannot_exceed_population():
    config = base_config()
    config["tournament_size"] = 10

    with pytest.raises(ValueError):
        GeneticOptimizer.optimize(
            cost_matrix=toy_matrix(),
            seed=42,
            budget=20,
            config=config,
        )