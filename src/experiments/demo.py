"""Demo de sustentación en un solo comando (Persona 5): python -m src.experiments.demo"""
import argparse
import time

from ..visualization import animation, map_routes, plots
from . import benchmark, run_tsp
from .run_tsp import leer_yaml

LIMITE_S = 120  # el enunciado pide una corrida de demostración de menos de dos minutos


def correr_demo(config="config/demo.yaml", out="results/demo"):
    """Benchmark corto, una corrida con history, mapa, GIF y figuras; devuelve los segundos totales."""
    start = time.perf_counter()
    exp = leer_yaml(config)["experiment"]
    v, seed, opt = float(exp["hourly_values_eur"][0]), exp["seeds"][0], exp["optimizers"][0]
    benchmark.main(["--config", config, "--out", out])
    # el benchmark no guarda history: el GIF necesita una corrida de run_tsp
    run_tsp.main(["--optimizer", opt, "--config", config, "--out", f"{out}/raw"])
    result = f"{out}/raw/{opt}_v{v:g}_seed{seed}.json"
    map_routes.main(["--results", result, "--out", f"{out}/figures"])
    animation.main(["--results", result, "--out", f"{out}/gifs"])
    plots.main(["--benchmark", f"{out}/raw/benchmark_runs.json", "--summary", f"{out}/summary/benchmark_summary.csv",
                "--hourly_value", f"{v:g}", "--out", f"{out}/figures"])
    return time.perf_counter() - start


def main(argv=None):
    parser = argparse.ArgumentParser(description="Demo completa: ACO y GA, mapa, GIF y figuras.")
    parser.add_argument("--config", default="config/demo.yaml")
    parser.add_argument("--out", default="results/demo")
    args = parser.parse_args(argv)

    elapsed = correr_demo(args.config, args.out)
    print(f"Demo completa en {elapsed:.1f} s. Salidas en {args.out}/")
    # avisa sin fallar: si la máquina de la sustentación es lenta, la demo ya terminó y sus salidas sirven
    if elapsed > LIMITE_S:
        print(f"AVISO: la demo tardó más de {LIMITE_S} s; reduce budget o seeds en {args.config}")


if __name__ == "__main__":
    main()
