"""GIF de la evolución de la mejor ruta, solo desde history (Persona 5): python -m src.visualization.animation"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .map_routes import NOMBRES, ROOT, cargar_ciudades, marcar_avisos, plt, preparar_mapa, trazar_ruta


def cargar_history(path):
    """Registro de run_tsp con su history; benchmark_runs.json no lo guarda y se rechaza."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "runs" in data:
        raise ValueError("benchmark_runs.json no guarda el history completo; anima un JSON de run_tsp "
                         "(python -m src.experiments.run_tsp ...)")
    if not data.get("history"):
        raise ValueError(f"{path} no trae history")
    return data


def seleccionar_pasos(history, max_frames):
    """Primer paso, pasos donde baja el costo y el paso final; si sobran, submuestreo parejo con ambos extremos."""
    if max_frames < 2:
        raise ValueError("max_frames debe ser >= 2 para conservar el primer y el último paso")
    steps = [h for i, h in enumerate(history) if i == 0 or h["best_cost"] < history[i - 1]["best_cost"]]
    if steps[-1] is not history[-1]:
        steps.append(history[-1])  # el último cuadro muestra todas las evaluaciones gastadas
    if len(steps) <= max_frames:
        return steps
    return [steps[i] for i in np.linspace(0, len(steps) - 1, max_frames).round().astype(int)]


def texto_cuadro(step, budget, k, n):
    return (f"evaluaciones {step['evaluations']:,} / {budget:,}".replace(",", " ")
            + f" | costo {step['best_cost']:.2f} € | cuadro {k} de {n}")


def animar(record, cities, fps=4, max_frames=40, hold_s=3.0, labels="names", dpi=80):
    """Cuadros RGB y duraciones en ms; el mapa se dibuja una vez y en cada cuadro solo se mueve la ruta."""
    steps = seleccionar_pasos(record["history"], max_frames)
    fig, ax = plt.subplots(figsize=(12 if labels == "ids" else 10, 8), dpi=dpi)
    preparar_mapa(ax, cities, labels)
    marcar_avisos(fig, record)
    ax.set_title(f"{NOMBRES.get(record['optimizer'], record['optimizer'].upper())} | "
                 f"v = {record['hourly_value_eur']:g} €/h | seed {record['seed']}\n"
                 "Evolución de la mejor ruta encontrada", fontsize=11)
    info = ax.text(0.01, 0.01, "", transform=ax.transAxes, fontsize=9, bbox={"facecolor": "white", "lw": 0})
    artists, frames = trazar_ruta(ax, steps[0]["best_route"], cities), []
    ax.legend(loc="lower right", fontsize=8)
    for k, step in enumerate(steps, 1):
        trazar_ruta(ax, step["best_route"], cities, artists)
        info.set_text(texto_cuadro(step, record["budget"], k, len(steps)))
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
    plt.close(fig)
    return frames, [round(1000 / fps)] * (len(frames) - 1) + [round(hold_s * 1000)]


def guardar_gif(frames, durations, path):
    # una sola paleta para todos los cuadros: así los colores no parpadean entre cuadros
    first = frames[0].quantize(colors=64)
    rest = [f.quantize(palette=first, dither=Image.Dither.NONE) for f in frames[1:]]
    path.parent.mkdir(parents=True, exist_ok=True)
    first.save(path, save_all=True, append_images=rest, duration=durations, loop=0, optimize=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="GIF de la evolución de la mejor ruta a partir del history.")
    parser.add_argument("--results", required=True, help="JSON de run_tsp (benchmark_runs.json no trae history)")
    parser.add_argument("--fps", type=float, default=4)
    parser.add_argument("--max_frames", type=int, default=40)
    parser.add_argument("--hold_s", type=float, default=3.0, help="segundos que se queda el último cuadro")
    parser.add_argument("--labels", choices=("names", "ids", "none"), default="names")
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument("--out", default="results/gifs")
    args = parser.parse_args(argv)

    record = cargar_history(ROOT / args.results)
    frames, durations = animar(record, cargar_ciudades(record), args.fps, args.max_frames, args.hold_s,
                               args.labels, args.dpi)
    path = guardar_gif(frames, durations, ROOT / args.out / f"{Path(args.results).stem}.gif")
    print(f"{path} | {len(frames)} cuadros | {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
