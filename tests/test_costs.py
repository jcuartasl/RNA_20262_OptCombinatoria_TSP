"""Pruebas de la matriz de costos C(v)=v*T+P+F. Vehículo y peajes de aquí son FICTICIOS."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.costs.cost_matrix import build_components, load_components, load_toll_matrix
from src.costs.vehicle import PendingDataError, Vehicle
from src.optimizers.common import route_cost, validate_cost_matrix

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"


def toy():
    dist = pd.read_csv(FIXTURES / "toy_5cities_distance_km.csv", index_col=0)
    time = pd.read_csv(FIXTURES / "toy_5cities_time_h.csv", index_col=0)
    return list(dist.index), dist.to_numpy(float), time.to_numpy(float)


def fake_vehicle():
    return Vehicle(
        model="Vehículo ficticio SOLO PARA TESTS", fuel_type="gasolina 95",
        consumption_l_per_100km=6.0, fuel_price_eur_per_l=1.5,
        consumption_source="prueba", fuel_price_source="prueba", fuel_price_date="2026-01-01",
    )


def fake_tolls(n, seed=0):
    P = np.random.default_rng(seed).uniform(0.0, 40.0, (n, n))
    np.fill_diagonal(P, 0.0)
    return P


@pytest.fixture
def parts():
    names, D, T = toy()
    return names, D, T, fake_tolls(len(names))


@pytest.fixture
def comps(parts):
    names, D, T, P = parts
    return build_components(T, D, P, fake_vehicle(), names)


def test_matches_definition(comps, parts):
    _, D, T, P = parts
    v = 17.5
    expected = v * T + P + D / 100.0 * 6.0 * 1.5
    np.testing.assert_allclose(comps.cost_matrix(v), expected)


def test_diagonal_zero_and_asymmetry_kept(comps):
    cm = comps.cost_matrix(15.0)
    assert np.all(np.diag(cm) == 0.0)
    assert not np.allclose(cm, cm.T)


def test_accepted_by_common_validator(comps):
    validate_cost_matrix(comps.cost_matrix(15.0))


def test_zero_hourly_value_gives_fixed_costs(comps):
    np.testing.assert_allclose(comps.cost_matrix(0.0), comps.toll_eur + comps.fuel_eur)


def test_arc_cost_monotone_in_hourly_value(comps):
    low, high = comps.cost_matrix(10.0), comps.cost_matrix(40.0)
    assert np.all(high >= low - 1e-12)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_hourly_value(comps, bad):
    with pytest.raises(ValueError):
        comps.cost_matrix(bad)


def test_units_are_hours_and_km(comps):
    off = ~np.eye(comps.n, dtype=bool)
    speeds = comps.distance_km[off] / comps.time_h[off]
    assert 30.0 <= speeds.min() and speeds.max() <= 140.0  # detecta minutos vs horas


def test_route_totals_reconstruct_cost(comps):
    route = (0, 2, 1, 4, 3)
    totals = comps.route_totals(route)
    v = 12.0
    assert totals.cost(v) == pytest.approx(route_cost(comps.cost_matrix(v), route))
    arcs = list(zip(route, route[1:] + route[:1]))  # incluye el arco de retorno
    assert totals.toll_eur == pytest.approx(sum(comps.toll_eur[i, j] for i, j in arcs))
    assert totals.time_h == pytest.approx(sum(comps.time_h[i, j] for i, j in arcs))


def test_invalid_route_rejected(comps):
    with pytest.raises(ValueError):
        comps.route_totals((0, 1, 1, 2, 3))


def test_missing_toll_is_error_not_zero(parts):
    names, D, T, P = parts
    P[0, 1] = np.nan
    with pytest.raises(PendingDataError):
        build_components(T, D, P, fake_vehicle(), names)


def test_provisional_zero_warns_and_flags(parts):
    names, D, T, P = parts
    P[0, 1] = np.nan
    with pytest.warns(UserWarning):
        c = build_components(T, D, P, fake_vehicle(), names, toll_policy="provisional_zero")
    assert c.tolls_complete is False
    assert c.toll_eur[0, 1] == 0.0


def test_negative_toll_rejected(parts):
    names, D, T, P = parts
    P[0, 1] = -1.0
    with pytest.raises(ValueError):
        build_components(T, D, P, fake_vehicle(), names)


def test_nan_time_rejected(parts):
    names, D, T, P = parts
    T[0, 1] = np.nan
    with pytest.raises(ValueError):
        build_components(T, D, P, fake_vehicle(), names)


def test_shape_mismatch_rejected(parts):
    names, D, T, P = parts
    with pytest.raises(ValueError):
        build_components(T[:4, :4], D, P, fake_vehicle(), names)


def test_unknown_toll_policy(parts):
    names, D, T, P = parts
    with pytest.raises(ValueError):
        build_components(T, D, P, fake_vehicle(), names, toll_policy="otra")


def test_toll_file_missing_or_empty_is_pending(tmp_path):
    names, *_ = toy()
    with pytest.raises(PendingDataError):
        load_toll_matrix(tmp_path, names)
    (tmp_path / "toll_eur.csv").write_text("")
    with pytest.raises(PendingDataError):
        load_toll_matrix(tmp_path, names)


def test_toll_file_roundtrip_and_label_check(tmp_path):
    names, *_ = toy()
    P = fake_tolls(len(names))
    pd.DataFrame(P, index=names, columns=names).to_csv(tmp_path / "toll_eur.csv")
    np.testing.assert_allclose(load_toll_matrix(tmp_path, names), P)
    with pytest.raises(ValueError):
        load_toll_matrix(tmp_path, list(reversed(names)))


def test_real_dataset_components():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            c = load_components(fake_vehicle())
    except PendingDataError:
        pytest.skip("toll_eur.csv aún incompleto (peajes PENDIENTES)")
    assert c.n == 47 and c.tolls_complete
    validate_cost_matrix(c.cost_matrix(15.0))
