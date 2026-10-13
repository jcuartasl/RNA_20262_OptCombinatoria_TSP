"""Mapa esquemático de una ruta sobre lon/lat (Persona 5): python -m src.visualization.map_routes --results ..."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventanas: funciona en consola, en tests y en un clon limpio
import matplotlib.pyplot as plt
import numpy as np

from ..data.road_matrix import load_cities

ROOT = Path(__file__).resolve().parents[2]
AVISO = "Segmentos rectos: representación esquemática, no la geometría de las carreteras."
NOMBRES = {"aco": "ACO", "genetic": "GA"}


def cargar_registro(path, optimizer=None, hourly_value=None, seed=None):
    """Registro de run_tsp tal cual, o una corrida de benchmark_runs.json junto con su encabezado."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "runs" not in data:
        return data
    runs = [r for r in data["runs"] if r["optimizer"] == optimizer and r["hourly_value_eur"] == hourly_value
            and (seed is None or r["seed"] == seed)]
    if not runs:
        raise ValueError(f"No hay corridas con optimizer={optimizer}, v={hourly_value}, seed={seed} en {path}")
    run = min(runs, key=lambda r: r["best_cost"])  # sin seed se dibuja la mejor corrida del grupo
    return {**{k: v for k, v in data.items() if k != "runs"}, **run, "best_route": run["route"]}


def cargar_ciudades(record):
    """cities.csv del dataset del registro; falla si el orden no coincide con el de la ruta."""
    cities = load_cities(str(ROOT / record["experiment_config"]["dataset"]["directory"]))
    if cities["name"].tolist() != record["city_names"]:
        raise ValueError("Las ciudades de cities.csv no coinciden con city_names del registro")
    return cities


def preparar_mapa(ax, cities, labels="names"):
    """Parte fija del mapa (ciudades, etiquetas, ejes y aviso); el GIF la dibuja una sola vez."""
    lon, lat, fig = cities["lon"].to_numpy(), cities["lat"].to_numpy(), ax.figure
    ax.scatter(lon, lat, s=14, color="#333333", zorder=3)
    if labels != "none":
        for i, (x, y, name) in enumerate(zip(lon, lat, cities["name"])):
            ax.annotate(name if labels == "names" else str(i), (x, y), xytext=(3, 3), textcoords="offset points",
                        fontsize=6.5, color="#333333")
    if labels == "ids":
        fig.subplots_adjust(right=0.8)
        fig.text(0.82, 0.5, "\n".join(f"{i:>2}  {n}" for i, n in enumerate(cities["name"])), va="center",
                 fontsize=6.5, family="monospace")
    # un grado de longitud mide cos(lat) veces un grado de latitud: sin esto España sale estirada
    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    ax.set_xlabel("Longitud (°)")
    ax.set_ylabel("Latitud (°)")
    ax.grid(alpha=0.3, lw=0.5)
    fig.subplots_adjust(bottom=0.09)
    fig.text(0.5, 0.02, AVISO, ha="center", fontsize=9, style="italic", color="#a33")


def trazar_ruta(ax, route, cities, artists=None):
    """Ciclo cerrado con flechas de sentido; con artists solo actualiza sus datos (cuadros del GIF)."""
    cycle = np.append(route, route[0])
    x, y = cities["lon"].to_numpy()[cycle], cities["lat"].to_numpy()[cycle]
    mid, d = np.c_[(x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2], np.c_[np.diff(x), np.diff(y)]
    d /= np.linalg.norm(d, axis=1, keepdims=True)  # flechas del mismo tamaño: solo indican el sentido
    if artists is None:
        return {"line": ax.plot(x, y, color="#1f77b4", lw=1.3, zorder=2, label="mejor ruta")[0],
                "arrows": ax.quiver(mid[:, 0], mid[:, 1], d[:, 0], d[:, 1], angles="xy", pivot="middle",
                                    scale=60, width=0.0025, headwidth=4, headlength=4, color="#1f77b4", zorder=2),
                "start": ax.plot(x[0], y[0], "o", ms=9, mfc="none", mec="#d62728", mew=2, zorder=4,
                                 label="ciudad inicial")[0]}
    artists["line"].set_data(x, y)
    artists["arrows"].set_offsets(mid)
    artists["arrows"].set_UVC(d[:, 0], d[:, 1])
    artists["start"].set_data([x[0]], [y[0]])
    return artists


def titulo(record):
    return (f"{NOMBRES.get(record['optimizer'], record['optimizer'].upper())} | v = {record['hourly_value_eur']:g} €/h"
            f" | costo = {record['best_cost']:.2f} € | seed {record['seed']}")


def avisos(record):
    """Marcas de que la corrida no vale como resultado final (mock o peajes provisionales)."""
    return [m for k, m in (("is_mock", "MOCK: SOLO PRUEBAS"), ("provisional_tolls", "PEAJES PROVISIONALES"))
            if record.get(k)]


def marcar_avisos(fig, record):
    if avisos(record):
        fig.suptitle(" | ".join(avisos(record)), color="#c00", fontweight="bold")


def dibujar_ruta(record, cities, labels="names"):
    """Figura completa con la mejor ruta del registro."""
    fig, ax = plt.subplots(figsize=(12 if labels == "ids" else 10, 8))
    preparar_mapa(ax, cities, labels)
    trazar_ruta(ax, record["best_route"], cities)
    ax.set_title(f"{titulo(record)}\nMejor ruta encontrada (no es un óptimo demostrado)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    marcar_avisos(fig, record)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mapa esquemático de la mejor ruta de una corrida.")
    parser.add_argument("--results", required=True, help="JSON de run_tsp o benchmark_runs.json")
    parser.add_argument("--optimizer", help="solo con benchmark_runs.json")
    parser.add_argument("--hourly_value", type=float, help="solo con benchmark_runs.json")
    parser.add_argument("--seed", type=int, help="solo con benchmark_runs.json; sin él, la corrida de menor costo")
    parser.add_argument("--labels", choices=("names", "ids", "none"), default="names")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args(argv)

    record = cargar_registro(ROOT / args.results, args.optimizer, args.hourly_value, args.seed)
    fig = dibujar_ruta(record, cargar_ciudades(record), args.labels)
    path = ROOT / args.out / f"route_{record['optimizer']}_v{record['hourly_value_eur']:g}_seed{record['seed']}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
