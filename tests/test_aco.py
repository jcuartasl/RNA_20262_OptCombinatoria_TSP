"""Pruebas de ACO (Ant System básico).

Ejecutar desde la raíz del proyecto:  python -m pytest tests/test_aco.py -v

Ajusta el bloque de importaciones y CONFIG_PATH a la estructura de tu repo.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
import yaml

# --- Ajusta a tu proyecto ---------------------------------------------------
from src.optimizers.aco import ACO, REQUIRED_PARAMETERS
from src.optimizers.common import route_cost

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "aco.yaml"
# ----------------------------------------------------------------------------

# Configuración completa y autocontenida: las pruebas no dependen del YAML real
# (salvo las de la sección "YAML", que verifican justamente ese archivo).
BASE = {
    "n_ants": 20,
    "alpha": 1.0,
    "beta": 3.0,
    "rho": 0.5,
    "q": 1.0,
    "eps": 1e-9,
}


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def cfg(**overrides):
    return {**BASE, **overrides}


def asymmetric_matrix(n: int, seed: int = 0) -> np.ndarray:
    matrix = np.random.default_rng(seed).integers(1, 50, (n, n)).astype(float)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def symmetric_matrix(n: int, seed: int = 0) -> np.ndarray:
    points = np.random.default_rng(seed).random((n, 2))
    return np.linalg.norm(points[:, None] - points[None], axis=2)


def all_routes(n: int):
    """Todas las rutas que empiezan en 0 (fuerza bruta, solo n pequeño)."""
    return [np.array((0,) + p) for p in itertools.permutations(range(1, n))]


def arcs(route) -> set[tuple[int, int]]:
    route = list(route)
    return {(route[i], route[(i + 1) % len(route)]) for i in range(len(route))}


def build(matrix, *, seed: int = 1, budget: int = 10_000, **overrides) -> ACO:
    """Construye el optimizador e inicializa su estado, sin ejecutar el ciclo.

    Según la base, el estado (rng, contadores, tau, eta) se crea en el
    constructor o recién en run(); reset() lo deja listo en ambos casos.
    """
    opt = ACO(matrix, seed, budget, cfg(**overrides))
    reset = getattr(opt, "reset", None)
    (reset if callable(reset) else opt._reset_state)()
    return opt


def run(matrix, *, seed: int = 1, budget: int = 1000, **overrides) -> dict:
    return ACO.optimize(matrix, seed, budget, cfg(**overrides))


def is_permutation(route, n: int) -> bool:
    return sorted(int(c) for c in route) == list(range(n))


@pytest.fixture(scope="module")
def small():
    """Instancia asimétrica de 8 ciudades con su óptimo por fuerza bruta."""
    matrix = asymmetric_matrix(8)
    optimum = min(route_cost(matrix, r) for r in all_routes(8))
    return matrix, optimum


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
def test_base_config_covers_all_required_parameters():
    assert set(BASE) == set(REQUIRED_PARAMETERS)


def test_params_are_parsed_and_stored_after_construction():
    params = build(asymmetric_matrix(5)).params
    assert params == BASE
    assert isinstance(params["n_ants"], int)


def test_params_are_available_after_a_reset():
    opt = build(asymmetric_matrix(5))
    opt._reset_state()
    assert opt.params == BASE


def test_missing_config_is_rejected():
    with pytest.raises(ValueError, match="Faltan parámetros"):
        ACO(asymmetric_matrix(5), 1, 100)
    with pytest.raises(ValueError, match="Faltan parámetros"):
        ACO(asymmetric_matrix(5), 1, 100, None)


def test_empty_config_lists_missing_parameters():
    with pytest.raises(ValueError, match="Faltan parámetros"):
        ACO(asymmetric_matrix(5), 1, 100, {})


@pytest.mark.parametrize("missing", REQUIRED_PARAMETERS)
def test_every_parameter_is_required(missing):
    config = {k: v for k, v in BASE.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        ACO(asymmetric_matrix(5), 1, 100, config)


@pytest.mark.parametrize("config", [42, [1, 2]])
def test_non_mapping_config_is_rejected(config):
    with pytest.raises(TypeError):  # el mensaje puede venir de la base o de ACO
        ACO(asymmetric_matrix(5), 1, 100, config)


def test_non_mapping_section_is_rejected():
    with pytest.raises(TypeError, match="sección"):
        ACO(asymmetric_matrix(5), 1, 100, {"aco": 5})


@pytest.mark.parametrize("name", ["foo", "variant", "ants", "evaporation", "elitist_weight"])
def test_unknown_parameters_are_rejected(name):
    with pytest.raises(ValueError, match="desconocidos"):
        ACO(asymmetric_matrix(5), 1, 100, cfg(**{name: 1}))


def test_underscore_keys_are_ignored_as_comments():
    opt = build(asymmetric_matrix(5), _nota="comentario")
    assert opt.params == BASE


def test_config_section_aco_is_supported():
    config = {"genetic": {"population_size": 10}, "aco": BASE}
    assert ACO.optimize(asymmetric_matrix(6), 1, 100, config)["evaluations"] == 100


@pytest.mark.parametrize(
    "name, value, error",
    [
        ("n_ants", 0, ValueError),
        ("n_ants", -3, ValueError),
        ("n_ants", 2.5, ValueError),
        ("n_ants", True, TypeError),
        ("n_ants", None, TypeError),
        ("n_ants", "abc", TypeError),
        ("alpha", -1, ValueError),
        ("beta", -0.1, ValueError),
        ("rho", 0, ValueError),
        ("rho", -0.2, ValueError),
        ("rho", 1.5, ValueError),
        ("rho", float("nan"), ValueError),
        ("rho", float("inf"), ValueError),
        ("q", 0, ValueError),
        ("q", -1, ValueError),
        ("eps", 0, ValueError),
        ("eps", -1e-9, ValueError),
        ("alpha", None, TypeError),
        ("beta", [1], TypeError),
    ],
)
def test_invalid_values_are_rejected(name, value, error):
    with pytest.raises(error):
        ACO(asymmetric_matrix(5), 1, 100, cfg(**{name: value}))


def test_boundary_values_are_accepted():
    opt = build(asymmetric_matrix(5), rho=1.0, alpha=0.0, beta=0.0, n_ants=1)
    assert opt.params["rho"] == 1.0 and opt.params["n_ants"] == 1


def test_numeric_strings_and_integers_are_accepted():
    # PyYAML entrega 1e-9 (sin punto) como texto.
    opt = build(asymmetric_matrix(5), eps="1e-9", rho="0.3", n_ants="10", q=2)
    assert opt.params["eps"] == 1e-9
    assert opt.params["rho"] == 0.3
    assert opt.params["n_ants"] == 10 and isinstance(opt.params["n_ants"], int)
    assert opt.params["q"] == 2.0 and isinstance(opt.params["q"], float)


def test_integer_valued_float_is_accepted_for_n_ants():
    assert build(asymmetric_matrix(5), n_ants=20.0).params["n_ants"] == 20


# --------------------------------------------------------------------------- #
# YAML
# --------------------------------------------------------------------------- #
def yaml_parameters() -> dict:
    """Parámetros del YAML real, sea plano o dentro de una sección 'aco:'."""
    assert CONFIG_PATH.exists(), f"no existe {CONFIG_PATH}"
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    section = data["aco"] if isinstance(data, dict) and "aco" in data else data
    return {k: v for k, v in section.items() if not str(k).startswith("_")}


def test_yaml_file_has_exactly_the_required_parameters():
    found, expected = set(yaml_parameters()), set(REQUIRED_PARAMETERS)
    assert found == expected, (
        f"faltan en el YAML: {sorted(expected - found)}; "
        f"sobran (el código no los reconoce): {sorted(found - expected)}"
    )


def test_yaml_file_values_are_valid():
    assert ACO(asymmetric_matrix(5), 1, 100, CONFIG_PATH).params


def test_runs_from_yaml_path_as_str_and_as_path():
    for path in (str(CONFIG_PATH), CONFIG_PATH):
        assert ACO.optimize(asymmetric_matrix(8), 1, 200, path)["evaluations"] == 200


def test_yaml_with_scientific_notation_without_dot(tmp_path):
    path = tmp_path / "aco.yaml"
    path.write_text(
        "n_ants: 10\nalpha: 1\nbeta: 3\nrho: 0.5\nq: 1\neps: 1e-9\n", encoding="utf-8"
    )
    assert ACO.optimize(asymmetric_matrix(6), 1, 100, path)["evaluations"] == 100


def test_yaml_with_section(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump({"aco": BASE, "genetic": {}}), encoding="utf-8")
    assert ACO.optimize(asymmetric_matrix(6), 1, 100, path)["evaluations"] == 100


def test_yaml_ignores_comment_keys(tmp_path):
    path = tmp_path / "aco.yaml"
    path.write_text(yaml.safe_dump({**BASE, "_nota": "x"}), encoding="utf-8")
    assert ACO.optimize(asymmetric_matrix(6), 1, 100, path)["evaluations"] == 100


def test_partial_and_empty_yaml_are_rejected(tmp_path):
    partial = tmp_path / "partial.yaml"
    partial.write_text("n_ants: 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Faltan parámetros"):
        ACO(asymmetric_matrix(6), 1, 100, partial)

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Faltan parámetros"):
        ACO(asymmetric_matrix(6), 1, 100, empty)


def test_missing_yaml_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ACO(asymmetric_matrix(6), 1, 100, tmp_path / "no_existe.yaml")


def test_yaml_and_dict_give_the_same_result(tmp_path):
    path = tmp_path / "aco.yaml"
    path.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    matrix = symmetric_matrix(12)
    assert ACO.optimize(matrix, 3, 400, path) == ACO.optimize(matrix, 3, 400, BASE)


# --------------------------------------------------------------------------- #
# Heurística y estado inicial
# --------------------------------------------------------------------------- #
def test_heuristic_is_normalized_with_zero_diagonal():
    eta = build(asymmetric_matrix(8), beta=1.0).eta
    assert np.all(np.diag(eta) == 0.0)
    off_diagonal = eta[~np.eye(8, dtype=bool)]
    assert off_diagonal.max() == pytest.approx(1.0)
    assert np.all(off_diagonal > 0.0)


def test_heuristic_prefers_cheaper_arcs():
    matrix = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 2.0], [4.0, 2.0, 0.0]])
    eta = build(matrix, beta=1.0).eta
    assert eta[0, 1] > eta[1, 2] > eta[0, 2]


def test_heuristic_applies_beta_as_exponent():
    matrix = asymmetric_matrix(8)
    eta1 = build(matrix, beta=1.0).eta
    eta3 = build(matrix, beta=3.0).eta
    np.testing.assert_allclose(eta3, eta1**3)


def test_heuristic_does_not_depend_on_cost_scale():
    matrix = asymmetric_matrix(8)
    np.testing.assert_allclose(
        build(matrix, beta=3.0).eta, build(matrix * 1000.0, beta=3.0).eta
    )


def test_heuristic_handles_zero_cost_arcs():
    matrix = asymmetric_matrix(6)
    matrix[0, 1] = 0.0
    eta = build(matrix, beta=1.0).eta
    assert np.all(np.isfinite(eta))
    assert eta[0, 1] == pytest.approx(1.0)  # el arco más barato queda en el máximo


def test_heuristic_gives_zero_visibility_to_forbidden_arcs():
    matrix = asymmetric_matrix(6)
    matrix[0, 1] = np.inf
    try:
        opt = build(matrix, beta=1.0)
    except ValueError:
        pytest.skip("la base rechaza matrices con inf")
    assert opt.eta[0, 1] == 0.0
    assert np.all(np.isfinite(opt.eta))


def test_nearest_neighbor_cost_matches_manual_greedy_route():
    matrix = asymmetric_matrix(9, seed=3)
    visited, current, total = {0}, 0, 0.0
    for _ in range(8):
        nxt = min((j for j in range(9) if j not in visited), key=lambda j: matrix[current, j])
        total += matrix[current, nxt]
        visited.add(nxt)
        current = nxt
    total += matrix[current, 0]
    assert build(matrix)._nearest_neighbor_cost() == pytest.approx(total)


def test_nearest_neighbor_cost_is_positive_when_all_costs_are_zero():
    assert build(np.zeros((4, 4)))._nearest_neighbor_cost() == 1.0


def test_nearest_neighbor_cost_does_not_consume_budget():
    opt = build(asymmetric_matrix(8), budget=50)
    opt._nearest_neighbor_cost()
    assert opt.remaining == 50


def test_initial_pheromone_is_n_ants_over_nearest_neighbor_cost():
    opt = build(asymmetric_matrix(8), n_ants=7)
    np.testing.assert_allclose(opt.tau, 7 / opt._nearest_neighbor_cost())
    assert opt.tau.shape == (8, 8)


def test_initial_pheromone_is_finite_for_degenerate_matrices():
    opt = build(np.zeros((5, 5)))
    assert np.all(np.isfinite(opt.tau)) and np.all(opt.tau > 0.0)


def test_reset_state_rebuilds_pheromone():
    opt = build(asymmetric_matrix(8))
    initial = opt.tau.copy()
    opt.tau *= 5.0
    opt._reset_state()
    np.testing.assert_allclose(opt.tau, initial)


# --------------------------------------------------------------------------- #
# Construcción de rutas
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [2, 5, 30])
def test_constructed_routes_are_valid_permutations(n):
    routes = build(asymmetric_matrix(n))._construct_routes(25)
    assert routes.shape == (25, n)
    assert all(is_permutation(r, n) for r in routes)


def test_construction_with_a_single_ant():
    routes = build(asymmetric_matrix(6))._construct_routes(1)
    assert routes.shape == (1, 6) and is_permutation(routes[0], 6)


def test_construction_covers_different_start_cities():
    routes = build(asymmetric_matrix(10))._construct_routes(200)
    assert len(set(routes[:, 0])) > 5


def test_construction_is_valid_with_extreme_alpha_and_beta():
    opt = build(symmetric_matrix(40), alpha=12.0, beta=25.0)
    assert all(is_permutation(r, 40) for r in opt._construct_routes(30))


def test_construction_falls_back_to_uniform_when_all_weights_vanish():
    opt = build(asymmetric_matrix(10))
    opt.eta[:] = 0.0  # simula underflow total
    assert all(is_permutation(r, 10) for r in opt._construct_routes(20))


def test_ants_follow_the_strongest_pheromone_trail():
    n = 6
    opt = build(asymmetric_matrix(n), alpha=1.0, beta=0.0)
    opt.tau[:] = 1e-12
    for i in range(n):
        opt.tau[i, (i + 1) % n] = 1.0
    for route in opt._construct_routes(30):
        assert all(route[i + 1] == (route[i] + 1) % n for i in range(n - 1))


def test_ants_follow_the_cheapest_arcs_when_beta_dominates():
    n = 6
    matrix = np.full((n, n), 100.0)
    np.fill_diagonal(matrix, 0.0)
    for i in range(n):
        matrix[i, (i + 1) % n] = 1.0  # ciclo barato
    opt = build(matrix, alpha=0.0, beta=30.0)
    for route in opt._construct_routes(30):
        assert all(route[i + 1] == (route[i] + 1) % n for i in range(n - 1))


def test_alpha_zero_ignores_pheromone():
    n = 6
    matrix = np.full((n, n), 100.0)
    np.fill_diagonal(matrix, 0.0)
    for i in range(n):
        matrix[i, (i + 1) % n] = 1.0
    opt = build(matrix, alpha=0.0, beta=30.0)
    opt.tau = np.random.default_rng(0).random((n, n)) + 0.1  # feromona arbitraria
    for route in opt._construct_routes(30):
        assert all(route[i + 1] == (route[i] + 1) % n for i in range(n - 1))


def test_construction_is_reproducible_for_the_same_seed():
    a = build(asymmetric_matrix(12), seed=7)._construct_routes(10)
    b = build(asymmetric_matrix(12), seed=7)._construct_routes(10)
    np.testing.assert_array_equal(a, b)


def test_construction_does_not_modify_pheromone_or_heuristic():
    opt = build(asymmetric_matrix(10))
    tau, eta = opt.tau.copy(), opt.eta.copy()
    opt._construct_routes(10)
    np.testing.assert_array_equal(opt.tau, tau)
    np.testing.assert_array_equal(opt.eta, eta)


# --------------------------------------------------------------------------- #
# Evaluación de rutas
# --------------------------------------------------------------------------- #
def test_evaluate_routes_returns_the_route_costs():
    matrix = asymmetric_matrix(8)
    opt = build(matrix, budget=100)
    routes = opt._construct_routes(5)
    evaluated, costs = opt._evaluate_routes(routes)
    np.testing.assert_array_equal(evaluated, routes)
    np.testing.assert_allclose(costs, [route_cost(matrix, r) for r in routes])
    assert opt.remaining == 95


def test_evaluate_routes_stops_when_the_budget_runs_out():
    opt = build(asymmetric_matrix(8), budget=3)
    routes = opt._construct_routes(5)
    evaluated, costs = opt._evaluate_routes(routes)
    assert len(evaluated) == 3 and len(costs) == 3
    assert opt.remaining == 0


def test_evaluate_routes_with_no_budget_returns_empty_arrays():
    opt = build(asymmetric_matrix(8), budget=2)
    opt._evaluate_routes(opt._construct_routes(2))  # agota el presupuesto
    evaluated, costs = opt._evaluate_routes(opt._construct_routes(3))
    assert evaluated.shape == (0, 8) and costs.shape == (0,)


# --------------------------------------------------------------------------- #
# Feromona: evaporación y depósito
# --------------------------------------------------------------------------- #
def test_evaporation_scales_pheromone():
    opt = build(asymmetric_matrix(6), rho=0.3)
    opt.tau = np.random.default_rng(0).random((6, 6)) + 0.1
    before = opt.tau.copy()
    opt._evaporate()
    np.testing.assert_allclose(opt.tau, before * 0.7)


def test_full_evaporation_clears_pheromone():
    opt = build(asymmetric_matrix(6), rho=1.0)
    opt._evaporate()
    assert np.all(opt.tau == 0.0)


def test_deposit_marks_every_arc_including_the_closing_one():
    n = 8
    opt = build(asymmetric_matrix(n), q=1.0)
    opt.tau[:] = 0.0
    route = np.random.default_rng(1).permutation(n)

    opt._deposit(route[None, :], np.array([4.0]))  # q / costo = 0.25

    expected = np.zeros((n, n))
    for arc in arcs(route):
        expected[arc] = 0.25
    np.testing.assert_allclose(opt.tau, expected)
    assert opt.tau[route[-1], route[0]] == pytest.approx(0.25)  # arco de cierre


def test_deposit_is_directed_even_on_symmetric_matrices():
    n = 8
    opt = build(symmetric_matrix(n))
    opt.tau[:] = 0.0
    route = np.arange(n)
    opt._deposit(route[None, :], np.array([2.0]))
    assert opt.tau[0, 1] > 0.0 and opt.tau[1, 0] == 0.0


def test_deposit_amount_is_q_over_cost():
    opt = build(asymmetric_matrix(6), q=3.0)
    opt.tau[:] = 0.0
    opt._deposit(np.arange(6)[None, :], np.array([2.0]))
    assert opt.tau[0, 1] == pytest.approx(1.5)


def test_deposits_of_several_ants_accumulate_on_shared_arcs():
    opt = build(asymmetric_matrix(6), q=1.0)
    opt.tau[:] = 0.0
    route = np.arange(6)
    opt._deposit(np.array([route, route]), np.array([4.0, 2.0]))  # 0.25 + 0.5
    assert opt.tau[0, 1] == pytest.approx(0.75)
    assert opt.tau.sum() == pytest.approx(6 * 0.75)


def test_deposit_is_finite_for_zero_cost_routes():
    opt = build(asymmetric_matrix(6), q=1.0, eps=1e-9)
    opt.tau[:] = 0.0
    opt._deposit(np.arange(6)[None, :], np.array([0.0]))
    assert np.all(np.isfinite(opt.tau))
    assert opt.tau[0, 1] == pytest.approx(1.0 / 1e-9)


def test_better_routes_deposit_more_pheromone():
    opt = build(asymmetric_matrix(6), q=1.0)
    opt.tau[:] = 0.0
    a, b = np.arange(6), np.array([0, 2, 4, 1, 3, 5])
    opt._deposit(np.array([a, b]), np.array([10.0, 100.0]))
    assert opt.tau[0, 1] > opt.tau[0, 2]


def test_update_pheromone_evaporates_then_deposits_for_all_ants():
    n, q, rho, eps = 8, 3.0, 0.4, 1e-9
    matrix = asymmetric_matrix(n)
    opt = build(matrix, q=q, rho=rho, eps=eps)
    opt.tau[:] = 1.0

    rng = np.random.default_rng(2)
    routes = np.array([rng.permutation(n) for _ in range(5)])
    costs = np.array([route_cost(matrix, r) for r in routes])

    opt._update_pheromone(routes, costs)

    expected_mass = (1 - rho) * n * n + n * np.sum(q / np.maximum(costs, eps))
    assert opt.tau.sum() == pytest.approx(expected_mass)


def test_update_pheromone_on_a_single_arc_matches_the_formula():
    n, q, rho = 6, 2.0, 0.25
    opt = build(asymmetric_matrix(n), q=q, rho=rho)
    opt.tau[:] = 1.0
    route = np.arange(n)
    opt._update_pheromone(route[None, :], np.array([10.0]))
    assert opt.tau[0, 1] == pytest.approx((1 - rho) * 1.0 + q / 10.0)  # arco usado
    assert opt.tau[0, 2] == pytest.approx((1 - rho) * 1.0)  # arco no usado


# --------------------------------------------------------------------------- #
# Ejecución completa
# --------------------------------------------------------------------------- #
SEEDS = range(10)


def test_finds_the_optimum_on_a_small_instance(small):
    """Ant System puede estancarse cerca del óptimo; con alguna semilla lo halla."""
    matrix, optimum = small
    costs = [run(matrix, seed=s, budget=3000)["best_cost"] for s in SEEDS]
    assert min(costs) == pytest.approx(optimum)
    assert max(costs) <= 1.05 * optimum


def test_result_is_consistent():
    matrix = symmetric_matrix(15)
    result = run(matrix, budget=500)
    assert is_permutation(result["best_route"], 15)
    assert route_cost(matrix, result["best_route"]) == pytest.approx(result["best_cost"])


def test_same_seed_gives_the_same_result():
    matrix = symmetric_matrix(15)
    assert run(matrix, seed=5) == run(matrix, seed=5)


def test_different_seeds_explore_differently():
    matrix = symmetric_matrix(25)
    routes = {tuple(run(matrix, seed=s, budget=60)["best_route"]) for s in range(5)}
    assert len(routes) > 1


def test_budget_is_respected_exactly():
    assert run(asymmetric_matrix(10), budget=1003)["evaluations"] == 1003  # no múltiplo


def test_exact_multiple_budget_counts_all_iterations():
    result = run(asymmetric_matrix(10), budget=1000, n_ants=20)
    assert result["evaluations"] == 1000 and result["iterations"] == 50


def test_partial_last_iteration_is_not_counted():
    result = run(asymmetric_matrix(10), budget=1003, n_ants=20)
    assert result["iterations"] == 50  # 1000 evaluaciones completas + 3 sueltas


def test_budget_smaller_than_one_iteration():
    result = run(asymmetric_matrix(10), budget=7, n_ants=20)
    assert result["evaluations"] == 7
    assert result["iterations"] == 0
    assert is_permutation(result["best_route"], 10)


def test_only_affordable_ants_are_constructed(monkeypatch):
    """En la última iteración no se construyen rutas que no se podrán evaluar."""
    requested = []
    original = ACO._construct_routes

    def spy(self, ants):
        requested.append(ants)
        return original(self, ants)

    monkeypatch.setattr(ACO, "_construct_routes", spy)
    run(asymmetric_matrix(8), budget=47, n_ants=20)
    assert requested == [20, 20, 7]


def test_single_ant_colony_runs():
    result = run(asymmetric_matrix(8), budget=100, n_ants=1)
    assert result["evaluations"] == 100 and result["iterations"] == 100


def test_nearest_neighbor_reference_does_not_consume_budget():
    assert run(asymmetric_matrix(10), budget=40, n_ants=20)["evaluations"] == 40


def test_more_budget_never_gives_a_worse_best():
    matrix = symmetric_matrix(20)
    short = run(matrix, seed=2, budget=200)["best_cost"]
    long = run(matrix, seed=2, budget=2000)["best_cost"]
    assert long <= short


def test_two_city_instance():
    two = ACO.optimize(np.array([[0.0, 1.0], [2.0, 0.0]]), 1, 20, BASE)
    assert two["best_cost"] == pytest.approx(3.0)
    assert is_permutation(two["best_route"], 2)


def test_single_city_instance():
    try:
        one = ACO.optimize(np.zeros((1, 1)), 1, 5, BASE)
    except ValueError:
        pytest.skip("la base exige al menos 2 ciudades")
    assert one["best_cost"] == pytest.approx(0.0)


def test_extreme_exponents_do_not_break_the_run():
    result = run(symmetric_matrix(30), budget=400, alpha=10.0, beta=20.0)
    assert np.isfinite(result["best_cost"])
    assert is_permutation(result["best_route"], 30)


def test_full_evaporation_still_runs():
    result = run(symmetric_matrix(12), budget=200, rho=1.0)
    assert np.isfinite(result["best_cost"])


def test_matrix_with_zero_cost_arcs():
    matrix = asymmetric_matrix(8)
    matrix[2, 3] = matrix[3, 4] = 0.0
    assert np.isfinite(run(matrix, budget=300)["best_cost"])


def test_matrix_with_forbidden_arcs():
    matrix = asymmetric_matrix(8)
    matrix[0, 1] = np.inf
    try:
        result = run(matrix, budget=500)
    except ValueError:
        pytest.skip("la base rechaza matrices con inf")
    assert np.isfinite(result["best_cost"])


def test_cost_matrix_is_not_modified():
    matrix = symmetric_matrix(12)
    original = matrix.copy()
    run(matrix, budget=300)
    np.testing.assert_array_equal(matrix, original)


def test_config_dict_is_not_modified():
    config = cfg()
    original = dict(config)
    ACO.optimize(symmetric_matrix(8), 1, 100, config)
    assert config == original


def test_pheromone_stays_positive_and_finite_after_a_run():
    opt = build(symmetric_matrix(15), budget=400, n_ants=10)
    opt._run()
    assert np.all(np.isfinite(opt.tau)) and np.all(opt.tau >= 0.0)


def test_pheromone_concentrates_on_the_best_route():
    matrix = asymmetric_matrix(8)
    opt = build(matrix, budget=2000, n_ants=20)
    opt._run()
    best = list(opt.best_route)
    on_best = np.mean([opt.tau[a] for a in arcs(best)])
    assert on_best > opt.tau[~np.eye(8, dtype=bool)].mean()


def test_beats_random_search():
    matrix = symmetric_matrix(30)
    rng = np.random.default_rng(0)
    random_mean = np.mean([route_cost(matrix, rng.permutation(30)) for _ in range(300)])
    assert run(matrix, budget=3000)["best_cost"] < 0.5 * random_mean


def test_learning_beats_pure_construction_without_pheromone_feedback():
    """Con rho=1 la feromona se borra cada iteración: el aprendizaje debe ayudar."""
    matrix = symmetric_matrix(30)
    with_memory = np.mean([run(matrix, seed=s, budget=3000, rho=0.1)["best_cost"] for s in range(4)])
    without_memory = np.mean([run(matrix, seed=s, budget=3000, rho=1.0)["best_cost"] for s in range(4)])
    assert with_memory <= without_memory * 1.02