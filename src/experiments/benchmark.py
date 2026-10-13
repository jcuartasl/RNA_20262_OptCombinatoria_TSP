"""Benchmark ACO vs GA (Persona 5): python -m src.experiments.benchmark --config config/base.yaml"""
import argparse
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from ..costs.sensitivity import sweep
from .export import guardar_json, metadatos_dataset
from .run_tsp import OPTIMIZERS, ROOT, cargar_componentes, leer_yaml


def curva_compacta(history):
    """Puntos [evaluations, best_cost] donde baja el best-so-far, más el último paso."""
    curve = [[h["evaluations"], h["best_cost"]] for i, h in enumerate(history)
             if i == 0 or h["best_cost"] < history[i - 1]["best_cost"]]
    if curve[-1][0] != history[-1]["evaluations"]:
        curve.append([history[-1]["evaluations"], history[-1]["best_cost"]])
    return curve


def correr_benchmark(optimizers, components, hourly_values, seeds, budget):
    """Barre (v, seed) con sweep para cada optimizador y añade a cada fila tiempo de cómputo y curva."""
    total, runs, start = len(optimizers) * len(hourly_values) * len(seeds), [], time.perf_counter()
    for name, (cls, algo_cfg) in optimizers.items():
        extras = []

        def envuelto(cm, seed, budget, config):
            v = hourly_values[len(extras) // len(seeds)]  # sweep recorre v por fuera y seeds por dentro
            t0 = time.perf_counter()
            res = cls.optimize(cm, seed, budget, config)
            extras.append({"elapsed_s": time.perf_counter() - t0, "iterations": res["iterations"],
                           "stop_reason": res["stop_reason"], "curve": curva_compacta(res["history"])})
            done, elapsed = len(runs) + len(extras), time.perf_counter() - start
            print(f"[{done:>{len(str(total))}}/{total}] {name:<7} v={v:<5g} seed={seed:<3} "
                  f"costo={res['best_cost']:.2f}  {extras[-1]['elapsed_s']:.1f} s | transcurrido "
                  f"{elapsed / 60:.1f} min  ETA ~{elapsed / done * (total - done) / 60:.0f} min", flush=True)
            return res

        rows = sweep(components, hourly_values, envuelto, seeds, budget, algo_cfg, name)
        runs += [{**row, "route": list(row["route"]), **extra} for row, extra in zip(rows, extras, strict=True)]
    return runs


def resumir(runs):
    """Estadísticas por optimizador y valor horario: costo, tiempo de cómputo y componentes medios."""
    g = pd.DataFrame(runs).groupby(["optimizer", "hourly_value_eur"], sort=False)
    out = g["best_cost"].agg(n_runs="count", mean_cost="mean", std_cost="std", min_cost="min", max_cost="max")
    out = out.join(g["elapsed_s"].agg(mean_elapsed_s="mean", std_elapsed_s="std"))
    return out.join(g[["time_h", "toll_eur", "fuel_eur"]].mean().add_prefix("mean_")).reset_index()


def datos_maquina():
    """elapsed_s depende de la máquina; se registra para poder comparar tiempos entre equipos."""
    return {"os": platform.platform(), "processor": platform.processor() or platform.machine(),
            "python": platform.python_version()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark ACO vs GA: mismas matrices y mismo budget por valor v.")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", help="reemplaza experiment.seeds del YAML")
    parser.add_argument("--hourly_values", type=float, nargs="+", help="reemplaza experiment.hourly_values_eur")
    parser.add_argument("--optimizers", nargs="+", choices=sorted(OPTIMIZERS))
    parser.add_argument("--budget", type=int)
    parser.add_argument("--toll_policy", choices=("error", "provisional_zero"))
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)

    cfg = leer_yaml(args.config)
    exp = cfg["experiment"]
    seeds = args.seeds or exp["seeds"]
    values = [float(v) for v in args.hourly_values or exp["hourly_values_eur"]]
    budget = exp["budget"] if args.budget is None else args.budget
    policy = args.toll_policy or cfg["tolls"]["policy"]
    optimizers = {n: (OPTIMIZERS[n][0], leer_yaml(OPTIMIZERS[n][1])) for n in args.optimizers or exp["optimizers"]}
    components = cargar_componentes(cfg, policy)
    runs = correr_benchmark(optimizers, components, values, seeds, budget)
    header = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "machine": datos_maquina(),
        "config_file": args.config, "experiment_config": cfg, "seeds": seeds, "hourly_values_eur": values,
        "budget": budget, "algo_configs": {n: c for n, (_, c) in optimizers.items()}, "toll_policy": policy,
        "provisional_tolls": policy == "provisional_zero", "tolls_complete": components.tolls_complete,
        "dataset": metadatos_dataset(ROOT / cfg["dataset"]["directory"]), "vehicle": asdict(components.vehicle),
        "city_names": list(components.city_names),
    }
    raw = guardar_json({**header, "runs": runs}, ROOT / args.out / "raw" / "benchmark_runs.json")
    summary = resumir(runs)
    (ROOT / args.out / "summary").mkdir(parents=True, exist_ok=True)
    summary.to_csv(ROOT / args.out / "summary" / "benchmark_summary.csv", index=False)
    print(summary.to_string(index=False, float_format="%.2f"))
    print(f"Corridas: {raw}\nResumen: {ROOT / args.out / 'summary' / 'benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
