"""Pruebas del vehículo y del combustible. Los valores de make() son FICTICIOS (solo tests)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.costs.vehicle import PendingDataError, Vehicle, fuel_cost_matrix, load_vehicle

CONFIG = Path(__file__).resolve().parents[1] / "config" / "sensitivity.yaml"


def make(**over):
    base = dict(
        model="Vehículo ficticio SOLO PARA TESTS",
        fuel_type="gasolina 95",
        consumption_l_per_100km=6.0,
        fuel_price_eur_per_l=1.5,
        consumption_source="fuente de prueba",
        fuel_price_source="fuente de prueba",
        fuel_price_date="2026-01-01",
    )
    base.update(over)
    return Vehicle(**base)


def test_fuel_cost_per_km():
    assert make().fuel_cost_per_km == pytest.approx(0.09)


def test_fuel_cost_matrix_formula():
    D = np.array([[0.0, 100.0], [200.0, 0.0]])
    F = fuel_cost_matrix(D, make())
    np.testing.assert_allclose(F, [[0.0, 9.0], [18.0, 0.0]])  # (D/100)*6*1.5


def test_fuel_cost_matrix_must_be_square():
    with pytest.raises(ValueError):
        fuel_cost_matrix(np.ones((2, 3)), make())


@pytest.mark.parametrize(
    "field",
    ["model", "fuel_type", "consumption_l_per_100km", "fuel_price_eur_per_l",
     "consumption_source", "fuel_price_source", "fuel_price_date"],
)
def test_pending_fields_raise(field):
    with pytest.raises(PendingDataError):
        make(**{field: None})
    if field not in ("consumption_l_per_100km", "fuel_price_eur_per_l"):
        with pytest.raises(PendingDataError):
            make(**{field: "PENDIENTE"})


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
def test_invalid_numbers_raise(bad):
    with pytest.raises(ValueError):
        make(consumption_l_per_100km=bad)
    with pytest.raises(ValueError):
        make(fuel_price_eur_per_l=bad)


def test_pending_error_is_value_error():
    assert issubclass(PendingDataError, ValueError)


def test_load_vehicle_from_dict_and_section():
    data = dict(
        model="X", fuel_type="diésel", consumption_l_per_100km=5.0, fuel_price_eur_per_l=1.4,
        consumption_source="s", fuel_price_source="s", fuel_price_date="2026-01-01",
    )
    assert load_vehicle(data) == load_vehicle({"vehicle": data})


def test_load_vehicle_rejects_unknown_keys():
    with pytest.raises(ValueError):
        load_vehicle({"vehicle": {"modelo_raro": 1}})


def test_yaml_structure():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert {"vehicle", "tolls", "sensitivity"} <= set(cfg)
    assert cfg["tolls"]["policy"] in ("error", "provisional_zero")
    assert len(cfg["sensitivity"]["hourly_values_eur"]) >= 2


def test_real_vehicle_in_yaml_is_complete():
    try:
        load_vehicle(CONFIG)
    except PendingDataError:
        pytest.skip("Vehículo aún PENDIENTE en config/sensitivity.yaml")
