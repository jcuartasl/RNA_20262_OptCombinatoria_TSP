"""Pruebas del runner. Dataset de 5 ciudades en tmp_path; el MockOptimizer solo existe aquí."""
import json
import shutil
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.costs.vehicle import PendingDataError
from src.experiments.run_tsp import cargar_componentes, correr, leer_yaml, main
from src.optimizers.common import BaseOptimizer, route_cost

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures"
VEHICLE = {"model": "Vehículo ficticio SOLO PARA TESTS", "fuel_type": "gasolina 95",
           "consumption_l_per_100km": 6.0, "fuel_price_eur_per_l": 1.5, "consumption_source": "prueba",
           "fuel_price_source": "prueba", "fuel_price_date": "2026-01-01"}


class MockOptimizer(BaseOptimizer):
    """SOLO PARA TESTS: permutaciones aleatorias por el evaluador común; nunca es un resultado final."""

    name = "mock"
    is_mock = True

    def _run(self):
        while True:  # termina cuando evaluate lanza BudgetExhausted
            for _ in range(5):
                self.evaluate(self.rng.permutation(self.n))
            self.end_iteration()


def crear_config(tmp_path, missing_toll=False):
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(FIXTURES / "toy_5cities.csv", data / "cities.csv")
    for name in ("distance_km", "time_h"):
        shutil.copy(FIXTURES / f"toy_5cities_{name}.csv", data / f"{name}.csv")
    names = pd.read_csv(data / "cities.csv")["name"].tolist()
    P = np.random.default_rng(0).uniform(0.0, 40.0, (5, 5))
    np.fill_diagonal(P, 0.0)
    if missing_toll:
        P[0, 1] = np.nan
    pd.DataFrame(P, index=names, columns=names).to_csv(data / "toll_eur.csv")
    (data / "metadata.json").write_text(json.dumps({"dataset_version": "toy", "dataset_hash_sha256_short": "abc",
                                                    "query_date_utc": "2026-01-01"}), encoding="utf-8")
    (tmp_path / "vehicle.yaml").write_text(yaml.safe_dump({"vehicle": VEHICLE}), encoding="utf-8")
    cfg = {"dataset": {"directory": str(data)}, "vehicle_config": str(tmp_path / "vehicle.yaml"),
           "tolls": {"policy": "error"},
           "experiment": {"hourly_values_eur": [15.0, 30.0], "seeds": [3, 4], "budget": 300,
                          "optimizers": ["aco", "genetic"]}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture
def sin_red(monkeypatch):
    def bloqueado(*args, **kwargs):
        raise AssertionError("El runner intentó usar la red")
    monkeypatch.setattr(socket, "socket", bloqueado)


def correr_mock(tmp_path, seed=0, budget=40):
    comps = cargar_componentes(leer_yaml(crear_config(tmp_path)), "error")
    return comps, correr(MockOptimizer, comps, 15.0, seed, budget, {}, "error", {"version": "toy"})


def test_correr_mock_registro_valido(tmp_path):
    comps, rec = correr_mock(tmp_path)
    assert rec["optimizer"] == "mock" and rec["is_mock"] is True
    assert sorted(rec["best_route"]) == list(range(5))
    assert rec["best_cost"] == pytest.approx(route_cost(comps.cost_matrix(15.0), rec["best_route"]))
    assert rec["totals"]["time_h"] * 15.0 + rec["totals"]["toll_eur"] + rec["totals"]["fuel_eur"] \
        == pytest.approx(rec["best_cost"])
    assert rec["evaluations"] == 40 and rec["stop_reason"] == "budget"
    costs = [h["best_cost"] for h in rec["history"]]
    assert costs == sorted(costs, reverse=True)
    json.dumps(rec)


def test_correr_reproducible(tmp_path):
    comps, a = correr_mock(tmp_path, seed=7)
    b = correr(MockOptimizer, comps, 15.0, 7, 40, {}, "error", {"version": "toy"})
    a.pop("elapsed_s"), b.pop("elapsed_s")
    assert a == b


def test_correr_rechaza_costo_inconsistente(tmp_path):
    class MockMentiroso(MockOptimizer):
        """SOLO PARA TESTS: reporta un costo que no corresponde a su ruta."""

        @classmethod
        def optimize(cls, cost_matrix, seed, budget, config=None):
            return {**super().optimize(cost_matrix, seed, budget, config), "best_cost": 0.0}

    comps = cargar_componentes(leer_yaml(crear_config(tmp_path)), "error")
    with pytest.raises(ValueError, match="recalculado"):
        correr(MockMentiroso, comps, 15.0, 0, 20, {}, "error", {})


@pytest.mark.parametrize("optimizer", ["aco", "genetic"])
def test_main_cli_sobrescribe_yaml(tmp_path, sin_red, optimizer):
    cfg = crear_config(tmp_path)
    main(["--optimizer", optimizer, "--config", str(cfg), "--seed", "7", "--budget", "120",
          "--hourly_value", "20", "--out", str(tmp_path / "raw")])
    rec = json.loads((tmp_path / "raw" / f"{optimizer}_v20_seed7.json").read_text(encoding="utf-8"))
    assert (rec["seed"], rec["budget"], rec["hourly_value_eur"]) == (7, 120, 20.0)
    assert rec["optimizer"] == optimizer and rec["is_mock"] is False
    assert rec["toll_policy"] == "error" and rec["provisional_tolls"] is False and rec["tolls_complete"] is True
    assert rec["dataset"] == {"version": "toy", "hash": "abc", "query_date": "2026-01-01"}
    assert rec["config"] and rec["experiment_config"]["experiment"]["budget"] == 300
    assert 0 < rec["evaluations"] <= 120 and rec["history"][-1]["best_route"] == rec["best_route"]


def test_main_usa_valores_del_yaml(tmp_path, sin_red):
    main(["--optimizer", "aco", "--config", str(crear_config(tmp_path)), "--out", str(tmp_path / "raw")])
    rec = json.loads((tmp_path / "raw" / "aco_v15_seed3.json").read_text(encoding="utf-8"))
    assert (rec["seed"], rec["budget"], rec["hourly_value_eur"]) == (3, 300, 15.0)


def test_peajes_pendientes(tmp_path, sin_red):
    cfg, out = crear_config(tmp_path, missing_toll=True), tmp_path / "raw"
    with pytest.raises(PendingDataError):
        main(["--optimizer", "aco", "--config", str(cfg), "--budget", "60", "--out", str(out)])
    with pytest.warns(UserWarning, match="PROVISIONAL"):
        main(["--optimizer", "aco", "--config", str(cfg), "--budget", "60", "--toll_policy", "provisional_zero",
              "--out", str(out)])
    rec = json.loads((out / "aco_v15_seed3.json").read_text(encoding="utf-8"))
    assert rec["toll_policy"] == "provisional_zero" and rec["provisional_tolls"] is True
    assert rec["tolls_complete"] is False
