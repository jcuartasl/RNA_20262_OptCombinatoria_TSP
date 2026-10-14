"""Pruebas de sensibilidad horaria. Datos ficticios; el óptimo exacto sale de fuerza bruta (5 ciudades)."""
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.costs.cost_matrix import build_components
from src.costs.sensitivity import (
    RouteLine, analyze_thresholds, best_per_value, canonical_route, crossover,
    line_for_route, lower_envelope, sweep, thresholds, write_sensitivity_csv,
)
from src.costs.vehicle import Vehicle
from src.optimizers.common import route_cost

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"
GRID = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]


@pytest.fixture
def comps():
    dist = pd.read_csv(FIXTURES / "toy_5cities_distance_km.csv", index_col=0)
    time = pd.read_csv(FIXTURES / "toy_5cities_time_h.csv", index_col=0)
    n = len(dist)
    P = np.random.default_rng(0).uniform(0.0, 40.0, (n, n))
    np.fill_diagonal(P, 0.0)
    veh = Vehicle(
        model="Vehículo ficticio SOLO PARA TESTS", fuel_type="gasolina 95",
        consumption_l_per_100km=6.0, fuel_price_eur_per_l=1.5,
        consumption_source="prueba", fuel_price_source="prueba", fuel_price_date="2026-01-01",
    )
    return build_components(time.to_numpy(float), dist.to_numpy(float), P, veh, list(dist.index))


def brute_force(cm, seed, budget, config):
    n = cm.shape[0]
    best = None
    for perm in itertools.permutations(range(1, n)):
        route = (0,) + perm
        cost = route_cost(cm, route)
        if best is None or cost < best[0] - 1e-12:
            best = (cost, route)
    return {"best_route": list(best[1]), "best_cost": best[0], "evaluations": math.factorial(n - 1)}


def env_cost(segments, v):
    for s in segments:
        if s.v_from - 1e-9 <= v <= s.v_to + 1e-9:
            return s.line.cost_at(v)
    raise AssertionError(f"v={v} fuera de los segmentos")


def test_canonical_route_rotation():
    assert canonical_route([3, 0, 2, 1]) == (0, 2, 1, 3)


def test_crossover_formula():
    a, b = RouteLine((0,), 10.0, 100.0), RouteLine((1,), 5.0, 130.0)
    v = crossover(a, b)
    assert v == pytest.approx(6.0)
    assert a.cost_at(v) == pytest.approx(b.cost_at(v))
    assert crossover(a, RouteLine((2,), 10.0, 5.0)) is None


def test_envelope_synthetic_skips_dominated_line():
    l1, l2, l3 = RouteLine((1,), 10, 0), RouteLine((2,), 5, 30), RouteLine((3,), 1, 100)
    dominated = RouteLine((4,), 7, 100)
    segs = lower_envelope([l1, l2, l3, dominated], 0.0, 50.0)
    assert [s.line for s in segs] == [l1, l2, l3]
    assert thresholds(segs) == pytest.approx([6.0, 17.5])
    assert segs[0].v_from == 0.0 and segs[-1].v_to == 50.0


def test_envelope_single_line_and_errors():
    only = RouteLine((1,), 3.0, 1.0)
    assert len(lower_envelope([only], 0.0, 10.0)) == 1
    with pytest.raises(ValueError):
        lower_envelope([], 0.0, 10.0)
    with pytest.raises(ValueError):
        lower_envelope([only], 5.0, 1.0)


def test_envelope_over_all_routes_matches_brute_force(comps):
    lines = [line_for_route(comps, (0,) + p) for p in itertools.permutations(range(1, comps.n))]
    segs = lower_envelope(lines, 0.0, 160.0)
    for v in GRID:
        expected = brute_force(comps.cost_matrix(v), 0, 0, None)["best_cost"]
        assert env_cost(segs, v) == pytest.approx(expected, rel=1e-9)
    if min(lines, key=lambda l: l.fixed_eur).route != min(lines, key=lambda l: l.time_h).route:
        assert len(segs) >= 2  # ruta barata en peajes != ruta rápida -> debe haber cambio


def test_sweep_rows_and_thresholds(comps):
    rows = sweep(comps, GRID, brute_force, seeds=[0, 1], budget=1, optimizer="exacto")
    assert len(rows) == len(GRID) * 2
    assert all(r["tolls_complete"] for r in rows)
    segs = analyze_thresholds(comps, rows, GRID[0], GRID[-1])
    for v, row in best_per_value(rows).items():
        assert env_cost(segs, v) == pytest.approx(row["best_cost"], rel=1e-9)


def test_sweep_detects_inconsistent_reported_cost(comps):
    def liar(cm, seed, budget, config):
        res = brute_force(cm, seed, budget, config)
        return {**res, "best_cost": res["best_cost"] + 1.0}

    with pytest.raises(ValueError):
        sweep(comps, [10.0], liar, seeds=[0], budget=1)


def test_write_csv(comps, tmp_path):
    rows = sweep(comps, [0.0, 10.0], brute_force, seeds=[0], budget=1, optimizer="exacto")
    out = tmp_path / "sub" / "sensitivity.csv"
    write_sensitivity_csv(rows, out)
    df = pd.read_csv(out)
    assert len(df) == 2 and df.loc[0, "route"].split()[0] == "0"
    with pytest.raises(ValueError):
        write_sensitivity_csv([], out)
