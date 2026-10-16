"""Una corrida de ACO o GA sobre el TSP (Persona 5): python -m src.experiments.run_tsp --optimizer aco"""
import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from ..costs.cost_matrix import load_components
from ..costs.vehicle import load_vehicle
from ..optimizers.aco import ACO
from ..optimizers.common import route_cost
from ..optimizers.genetic import GeneticOptimizer
from .export import NOTA_TOLLS_COMPLETE, guardar_json, hashes_entradas, metadatos_dataset

ROOT = Path(__file__).resolve().parents[2]
OPTIMIZERS = {"aco": (ACO, "config/aco.yaml"), "genetic": (GeneticOptimizer, "config/genetic.yaml")}


def leer_yaml(path):
    with open(ROOT / path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def cargar_componentes(cfg, toll_policy):
    """Componentes de costo offline: dataset de cfg['dataset'] y vehículo de cfg['vehicle_config']."""
    vehicle = load_vehicle(leer_yaml(cfg["vehicle_config"]))
    return load_components(vehicle, ROOT / cfg["dataset"]["directory"], toll_policy=toll_policy)


def correr(optimizer_cls, components, hourly_value, seed, budget, algo_config, toll_policy, dataset):
    """Corre optimize una vez, verifica el costo reportado y arma el registro de la corrida."""
    cm = components.cost_matrix(hourly_value)
    start = time.perf_counter()
    res = optimizer_cls.optimize(cm, seed, budget, algo_config)
    elapsed = time.perf_counter() - start
    recomputed = route_cost(cm, res["best_route"])
    if not np.isclose(recomputed, res["best_cost"], rtol=1e-9, atol=1e-6):
        raise ValueError(f"Costo reportado {res['best_cost']} != recalculado {recomputed} ({optimizer_cls.name})")
    return {
        "optimizer": optimizer_cls.name, "is_mock": getattr(optimizer_cls, "is_mock", False),
        "hourly_value_eur": float(hourly_value), "toll_policy": toll_policy,
        # provisional_zero rellena peajes faltantes con 0: nunca vale como resultado final
        "provisional_tolls": toll_policy == "provisional_zero", "tolls_complete": components.tolls_complete,
        "tolls_complete_note": NOTA_TOLLS_COMPLETE, "dataset": dataset, "vehicle": asdict(components.vehicle), "city_names": list(components.city_names),
        **res, "totals": asdict(components.route_totals(res["best_route"])), "elapsed_s": elapsed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Una corrida de ACO o GA sobre el TSP de las 47 capitales.")
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZERS), required=True)
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--hourly_value", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--toll_policy", choices=("error", "provisional_zero"))
    parser.add_argument("--algo_config", help="YAML de parámetros del optimizador (por defecto config/<optimizer>.yaml)")
    parser.add_argument("--out", default="results/raw")
    args = parser.parse_args(argv)

    cfg = leer_yaml(args.config)
    exp = cfg["experiment"]
    cls, algo_path = OPTIMIZERS[args.optimizer]
    # el CLI manda sobre el YAML; sin flag se usa el primer valor de cada lista
    v = exp["hourly_values_eur"][0] if args.hourly_value is None else args.hourly_value
    seed = exp["seeds"][0] if args.seed is None else args.seed
    budget = exp["budget"] if args.budget is None else args.budget
    policy = args.toll_policy or cfg["tolls"]["policy"]
    algo_file = args.algo_config or algo_path
    rec = correr(cls, cargar_componentes(cfg, policy), v, seed, budget, leer_yaml(algo_file),
                 policy, metadatos_dataset(ROOT / cfg["dataset"]["directory"]))
    # los YAML de ambos optimizadores, más el --algo_config usado si es otro archivo
    algo_files = list(dict.fromkeys([path for _, path in OPTIMIZERS.values()] + [algo_file]))
    rec.update(config_file=args.config, experiment_config=cfg, algo_config_file=algo_file,
               input_hashes=hashes_entradas(ROOT, cfg, algo_files))
    path = guardar_json(rec, ROOT / args.out / f"{args.optimizer}_v{float(v):g}_seed{seed}.json")
    print(f"{args.optimizer} | v={float(v):g} EUR/h | seed={seed} | costo={rec['best_cost']:.2f} EUR "
          f"| evaluaciones={rec['evaluations']} | {rec['elapsed_s']:.1f} s | {path}")


if __name__ == "__main__":
    main()
