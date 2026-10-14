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
from src.experiments import benchmark
from src.experiments.run_tsp import cargar_componentes, correr, leer_yaml, main
from src.data.road_matrix import load_cities
from src.optimizers.common import BaseOptimizer, route_cost
from src.visualization import animation, map_routes, plots

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


def test_curva_compacta():
    costs = [100.0, 100.0, 90.0, 90.0, 80.0, 80.0]
    history = [{"evaluations": 10 * (i + 1), "best_cost": c} for i, c in enumerate(costs)]
    assert benchmark.curva_compacta(history) == [[10, 100.0], [30, 90.0], [50, 80.0], [60, 80.0]]


def test_benchmark_misma_matriz_y_budget(tmp_path):
    vistos = []

    class Espia(MockOptimizer):
        """SOLO PARA TESTS: registra la matriz y el budget que recibe."""

        @classmethod
        def optimize(cls, cost_matrix, seed, budget, config=None):
            vistos.append((cls.name, cost_matrix.copy(), budget))
            return super().optimize(cost_matrix, seed, budget, config)

    a, b = type("EspiaA", (Espia,), {"name": "a"}), type("EspiaB", (Espia,), {"name": "b"})
    comps = cargar_componentes(leer_yaml(crear_config(tmp_path)), "error")
    runs = benchmark.correr_benchmark({"a": (a, {}), "b": (b, {})}, comps, [10.0, 40.0], [0, 1], 30)
    assert len(runs) == len(vistos) == 8
    for (_, cm_a, budget_a), (_, cm_b, budget_b) in zip(vistos[:4], vistos[4:]):
        assert np.array_equal(cm_a, cm_b) and budget_a == budget_b == 30


def test_benchmark_flags_curva_y_resumen(tmp_path, sin_red):
    benchmark.main(["--config", str(crear_config(tmp_path)), "--seeds", "0", "1", "--hourly_values", "10", "40",
                    "--budget", "150", "--out", str(tmp_path)])
    data = json.loads((tmp_path / "raw" / "benchmark_runs.json").read_text(encoding="utf-8"))
    assert (data["seeds"], data["hourly_values_eur"], data["budget"]) == ([0, 1], [10.0, 40.0], 150)
    assert set(data["machine"]) == {"os", "processor", "python"} and data["dataset"]["version"] == "toy"
    assert data["toll_policy"] == "error" and data["provisional_tolls"] is False
    runs = data["runs"]
    assert sorted((r["optimizer"], r["hourly_value_eur"], r["seed"]) for r in runs) == sorted(
        (o, v, s) for o in ("aco", "genetic") for v in (10.0, 40.0) for s in (0, 1))
    for r in runs:
        assert "history" not in r and 0 < r["evaluations"] <= 150 and sorted(r["route"]) == list(range(5))
        evals, costs = zip(*r["curve"])
        assert list(evals) == sorted(evals) and all(x >= y for x, y in zip(costs, costs[1:]))
        # si el zip de sweep con las curvas se desalineara, la curva no terminaría en el best_cost de su fila
        assert r["curve"][-1] == [r["evaluations"], r["best_cost"]]
    summary = pd.read_csv(tmp_path / "summary" / "benchmark_summary.csv")
    assert len(summary) == 4 and (summary["n_runs"] == 2).all()
    for _, s in summary.iterrows():
        costs = [r["best_cost"] for r in runs
                 if (r["optimizer"], r["hourly_value_eur"]) == (s["optimizer"], s["hourly_value_eur"])]
        assert (s["mean_cost"], s["min_cost"], s["max_cost"]) == pytest.approx(
            (np.mean(costs), min(costs), max(costs)))
        assert s["std_cost"] == pytest.approx(np.std(costs, ddof=1))
        group = [r for r in runs if (r["optimizer"], r["hourly_value_eur"]) == (s["optimizer"], s["hourly_value_eur"])]
        for col in ("time_h", "toll_eur", "fuel_eur"):
            assert s[f"std_{col}"] == pytest.approx(np.std([r[col] for r in group], ddof=1))


def test_mapa_desde_run_tsp(tmp_path, sin_red):
    main(["--optimizer", "aco", "--config", str(crear_config(tmp_path)), "--budget", "60",
          "--out", str(tmp_path / "raw")])
    map_routes.main(["--results", str(tmp_path / "raw" / "aco_v15_seed3.json"), "--out", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "route_aco_v15_seed3.png").stat().st_size > 0


def test_figura_titulo_aviso_y_ciclo(tmp_path):
    _, rec = correr_mock(tmp_path)
    fig = map_routes.dibujar_ruta(rec, load_cities(str(tmp_path / "data")))
    ax = fig.axes[0]
    assert "v = 15 €/h" in ax.get_title() and f"{rec['best_cost']:.2f} €" in ax.get_title()
    assert map_routes.AVISO in [t.get_text() for t in fig.texts]
    assert "MOCK" in fig._suptitle.get_text()
    xs, ys = ax.lines[0].get_data()
    assert len(xs) == 6 and (xs[0], ys[0]) == (xs[-1], ys[-1])


def test_etiquetas_ids_con_nombres_al_margen(tmp_path):
    _, rec = correr_mock(tmp_path)
    fig = map_routes.dibujar_ruta(rec, load_cities(str(tmp_path / "data")), labels="ids")
    assert [t.get_text() for t in fig.axes[0].texts] == ["0", "1", "2", "3", "4"]
    assert any("Madrid" in t.get_text() for t in fig.texts)


def test_trazar_ruta_actualiza_sin_crear_artistas(tmp_path):
    crear_config(tmp_path)
    cities = load_cities(str(tmp_path / "data"))
    fig = map_routes.plt.figure()
    ax = fig.add_subplot()
    artists = map_routes.trazar_ruta(ax, [0, 1, 2, 3, 4], cities)
    n_lines = len(ax.lines)
    assert map_routes.trazar_ruta(ax, [0, 2, 4, 1, 3], cities, artists) is artists and len(ax.lines) == n_lines
    assert list(artists["line"].get_xdata()) == cities["lon"].to_numpy()[[0, 2, 4, 1, 3, 0]].tolist()


def test_mapa_elige_corrida_del_benchmark(tmp_path, sin_red):
    benchmark.main(["--config", str(crear_config(tmp_path)), "--seeds", "0", "1", "--hourly_values", "10",
                    "--budget", "150", "--out", str(tmp_path)])
    path = tmp_path / "raw" / "benchmark_runs.json"
    rec = map_routes.cargar_registro(path, "genetic", 10.0, 1)
    assert (rec["optimizer"], rec["seed"]) == ("genetic", 1) and rec["best_route"] == rec["route"]
    runs = [r for r in json.loads(path.read_text(encoding="utf-8"))["runs"] if r["optimizer"] == "genetic"]
    assert map_routes.cargar_registro(path, "genetic", 10.0)["best_cost"] == min(r["best_cost"] for r in runs)
    with pytest.raises(ValueError, match="No hay corridas"):
        map_routes.cargar_registro(path, "aco", 99.0)


def test_mapa_rechaza_ciudades_distintas(tmp_path):
    _, rec = correr_mock(tmp_path)
    rec = {**rec, "experiment_config": {"dataset": {"directory": str(tmp_path / "data")}},
           "city_names": rec["city_names"][::-1]}
    with pytest.raises(ValueError, match="no coinciden"):
        map_routes.cargar_ciudades(rec)


def prohibir_optimizar(monkeypatch):
    """Se llama después de generar el JSON: desde ahí, cualquier optimize hace fallar el test."""
    monkeypatch.setattr(BaseOptimizer, "optimize", classmethod(lambda *a, **k: pytest.fail("reoptimizó")))


def history_sintetico(costs):
    return [{"step": i, "evaluations": 10 * (i + 1), "best_cost": c, "best_route": [0, 1, 2, 3, 4]}
            for i, c in enumerate(costs)]


def test_seleccionar_pasos_mejoras_y_paso_final():
    h = history_sintetico([100.0, 100.0, 90.0, 80.0, 80.0])
    assert [s["step"] for s in animation.seleccionar_pasos(h, 40)] == [0, 2, 3, 4]


def test_seleccionar_pasos_submuestrea_con_extremos():
    h = history_sintetico([100.0 - i for i in range(30)])
    steps = animation.seleccionar_pasos(h, 7)
    assert len(steps) == 7 and steps[0] is h[0] and steps[-1] is h[-1]
    assert len({s["step"] for s in steps}) == 7
    with pytest.raises(ValueError):
        animation.seleccionar_pasos(h, 1)


def test_texto_cuadro():
    step = {"evaluations": 12345, "best_cost": 1830.614}
    assert animation.texto_cuadro(step, 50000, 3, 9) == "evaluaciones 12 345 / 50 000 | costo 1830.61 € | cuadro 3 de 9"


def test_gif_desde_history_sin_reoptimizar(tmp_path, sin_red, monkeypatch):
    main(["--optimizer", "aco", "--config", str(crear_config(tmp_path)), "--budget", "200",
          "--out", str(tmp_path / "raw")])
    prohibir_optimizar(monkeypatch)
    src = tmp_path / "raw" / "aco_v15_seed3.json"
    rec = json.loads(src.read_text(encoding="utf-8"))
    n = len(animation.seleccionar_pasos(rec["history"], 40))
    animation.main(["--results", str(src), "--fps", "5", "--hold_s", "2", "--out", str(tmp_path / "gifs")])
    with animation.Image.open(tmp_path / "gifs" / "aco_v15_seed3.gif") as gif:
        assert gif.n_frames == n
        durations = []
        for i in range(gif.n_frames):
            gif.seek(i)
            durations.append(gif.info["duration"])
    assert durations[-1] == 2000 and set(durations[:-1]) <= {200}


def test_gif_submuestrea_hasta_max_frames(tmp_path, sin_red, monkeypatch):
    _, rec = correr_mock(tmp_path)
    prohibir_optimizar(monkeypatch)
    rec = {**rec, "history": history_sintetico([100.0 - i for i in range(30)])}  # 30 mejoras
    frames, durations = animation.animar(rec, load_cities(str(tmp_path / "data")), max_frames=3, hold_s=1.5)
    assert len(frames) == len(durations) == 3 and durations[-1] == 1500
    assert frames[0].size == (800, 640)


def test_gif_rechaza_benchmark_y_json_sin_history(tmp_path, sin_red):
    benchmark.main(["--config", str(crear_config(tmp_path)), "--seeds", "0", "--hourly_values", "10",
                    "--budget", "100", "--out", str(tmp_path)])
    with pytest.raises(ValueError, match="benchmark_runs.json no guarda el history"):
        animation.cargar_history(tmp_path / "raw" / "benchmark_runs.json")
    (tmp_path / "vacio.json").write_text(json.dumps({"best_route": [0, 1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="no trae history"):
        animation.cargar_history(tmp_path / "vacio.json")


def benchmark_sintetico(tmp_path, n_seeds, values=(10.0, 40.0), budget=100):
    """benchmark_runs.json y summary ficticios con costos conocidos (SOLO PARA TESTS)."""
    runs = []
    for opt in ("aco", "genetic"):
        for v in values:
            for s in range(n_seeds):
                t, toll, fuel = 10.0 + s, 5.0, 20.0
                cost = v * t + toll + fuel
                runs.append({"optimizer": opt, "hourly_value_eur": v, "seed": s, "best_cost": cost, "time_h": t,
                             "toll_eur": toll, "fuel_eur": fuel, "evaluations": budget, "route": [0, 1, 2, 3, 4],
                             "elapsed_s": 0.1, "curve": [[5 + s, cost + 50], [20 + s, cost + 10], [budget, cost]]})
    path, summary = tmp_path / f"runs_{n_seeds}.json", tmp_path / f"summary_{n_seeds}.csv"
    path.write_text(json.dumps({"budget": budget, "hourly_values_eur": list(values), "provisional_tolls": False,
                                "runs": runs}), encoding="utf-8")
    benchmark.resumir(runs).to_csv(summary, index=False)
    return path, summary


def test_alinear_curvas_escalonadas():
    x, y = plots.alinear_curvas([[[1, 10.0], [5, 6.0], [10, 6.0]], [[2, 9.0], [4, 7.0], [10, 3.0]]], 10)
    assert x.tolist() == [2, 4, 5, 10]
    assert y.tolist() == [[10.0, 10.0, 6.0, 6.0], [9.0, 7.0, 7.0, 3.0]]


def test_tipo_banda_segun_seeds():
    assert [plots.tipo_banda(n) for n in (1, 3, 6)] == [None, "min-max", "p25-p75"]


@pytest.mark.parametrize("n_seeds", [2, 6])
def test_figuras_nombres_titulos_y_pocas_seeds(tmp_path, sin_red, n_seeds):
    path, summary = benchmark_sintetico(tmp_path, n_seeds)
    plots.main(["--benchmark", str(path), "--summary", str(summary), "--hourly_value", "40", "--out",
                str(tmp_path / "fig")])
    for name in ("convergence_v40.png", "final_cost_boxplot.png", "cost_vs_v.png", "cost_breakdown_vs_v.png",
                 "cost_diff_vs_v.png"):
        assert (tmp_path / "fig" / name).stat().st_size > 0
    data, runs, summ = plots.cargar_benchmark(path, summary)
    figs = [plots.figura_convergencia(data, runs, 40.0), plots.figura_cajas(data, runs),
            plots.figura_costo_v(data, summ), plots.figura_desglose(data, summ), plots.figura_diferencia(data, summ)]
    for fig in figs:
        title = fig._suptitle.get_text()
        assert f"n = {n_seeds} seeds" in title and ("ilustrativo" in title) == (n_seeds < 5)
        assert all("(" in ax.get_ylabel() for ax in fig.axes if ax.get_visible() and ax.get_ylabel())
        # la diferencia es una sola serie: el título la nombra y no lleva leyenda
        assert fig is figs[-1] or fig.legends or any(ax.get_legend() for ax in fig.axes)
    legend = [t.get_text() for t in figs[0].axes[0].get_legend().get_texts()]
    assert any(("percentiles 25–75" if n_seeds >= 5 else "mín–máx") in t for t in legend)
    assert all((len(ax.lines) > 0) == (n_seeds >= 5) for ax in figs[1].axes if ax.get_visible())
    for fig in figs:
        plots.plt.close(fig)


def test_figuras_rechazan_datos_inconsistentes(tmp_path):
    path2, summary2 = benchmark_sintetico(tmp_path, 2)
    _, summary6 = benchmark_sintetico(tmp_path, 6)
    with pytest.raises(ValueError, match="no corresponde"):
        plots.cargar_benchmark(path2, summary6)
    bad = pd.read_csv(summary2)
    bad.loc[0, "mean_cost"] += 1.0
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="no cumple"):
        plots.cargar_benchmark(path2, tmp_path / "bad.csv")
    data, runs, _ = plots.cargar_benchmark(path2, summary2)
    with pytest.raises(ValueError, match="No hay corridas con v=99"):
        plots.figura_convergencia(data, runs, 99.0)


def test_diferencia_medias_no_emparejada(tmp_path):
    _, summary = benchmark_sintetico(tmp_path, 3)
    summ = pd.read_csv(summary)
    ga = summ["optimizer"] == "genetic"
    summ.loc[ga, "mean_cost"] -= 30.0
    summ.loc[ga, "std_cost"] *= 2
    d = plots.diferencia_medias(summ).set_index("hourly_value_eur")
    aco, gen = summ[~ga].set_index("hourly_value_eur"), summ[ga].set_index("hourly_value_eur")
    assert d["diff"].tolist() == pytest.approx([-30.0, -30.0])
    assert d["se"].tolist() == pytest.approx(np.sqrt(aco["std_cost"] ** 2 / 3 + gen["std_cost"] ** 2 / 3).tolist())


def test_desglose_recorta_barras_en_cero(tmp_path):
    path, summary = benchmark_sintetico(tmp_path, 3)
    data, runs, summ = plots.cargar_benchmark(path, summary)
    summ["std_toll_eur"] = 50.0  # desviación mayor que la media de peajes (5 €)
    fig = plots.figura_desglose(data, summ)
    for c in fig.axes[1].containers:
        assert min(seg[:, 1].min() for seg in c.lines[2][0].get_segments()) >= 0
        assert max(seg[:, 1].max() for seg in c.lines[2][0].get_segments()) == pytest.approx(55.0)
    assert "recortan en 0" in fig._supxlabel.get_text()
    plots.plt.close(fig)
