"""Figuras del benchmark ACO vs GA (Persona 5): python -m src.visualization.plots --benchmark ..."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .map_routes import NOMBRES, ROOT, avisos, plt

COLORES = {"aco": "#2a78d6", "genetic": "#eb6834"}  # validados para daltonismo (ΔE ≥ 24 entre ambos)
TEXTO, GRIS = "#0b0b0b", "#52514e"
MIN_SEEDS = 5  # por debajo, cajas y percentiles no tienen sentido estadístico


def cargar_benchmark(path, summary_path):
    """Corridas y resumen; falla si el summary no corresponde al JSON o no cumple v·T + P + F."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    runs, summary = pd.DataFrame(data["runs"]), pd.read_csv(summary_path)
    groups = runs.groupby(["optimizer", "hourly_value_eur"]).size().rename("n").to_frame().join(
        summary.set_index(["optimizer", "hourly_value_eur"])["n_runs"], how="outer")
    if groups.isna().any().any() or (groups["n"] != groups["n_runs"]).any():
        raise ValueError(f"{summary_path} no corresponde a {path}: grupos o n_runs distintos")
    rebuilt = summary["hourly_value_eur"] * summary["mean_time_h"] + summary["mean_toll_eur"] + summary["mean_fuel_eur"]
    if not np.allclose(rebuilt, summary["mean_cost"], rtol=1e-9, atol=1e-6):
        raise ValueError("El summary no cumple mean_cost = v·mean_time_h + mean_toll_eur + mean_fuel_eur")
    return data, runs, summary


def tipo_banda(n):
    return "p25-p75" if n >= MIN_SEEDS else "min-max" if n >= 2 else None


def alinear_curvas(curves, budget):
    """Eje x común y matriz seeds × x con el best-so-far escalonado de cada seed."""
    x = np.unique(np.concatenate([np.asarray(c)[:, 0] for c in curves] + [[budget]]))
    x = x[x >= max(c[0][0] for c in curves)]  # desde donde todas las seeds ya tienen un costo
    return x, np.array([np.asarray(c)[np.searchsorted(np.asarray(c)[:, 0], x, side="right") - 1, 1] for c in curves])


def nota_seeds(n):
    return f"n = {n} seeds por algoritmo" + (f" (n < {MIN_SEEDS}: ilustrativo, sin validez estadística)"
                                               if n < MIN_SEEDS else "")


def titular(fig, texto, data):
    fig.suptitle(texto, fontsize=12, color=TEXTO)
    if avisos(data):
        fig.text(0.01, 0.995, " | ".join(avisos(data)), va="top", color="#c00", fontweight="bold")


def estilo(ax):
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=GRIS)


def figura_convergencia(data, runs, v):
    sel = runs[runs["hourly_value_eur"] == v]
    if sel.empty:
        raise ValueError(f"No hay corridas con v={v:g} en el benchmark (hay {sorted(runs['hourly_value_eur'].unique())})")
    n = int(sel.groupby("optimizer").size().min())
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    for opt, g in sel.groupby("optimizer", sort=False):
        x, y = alinear_curvas(g["curve"].tolist(), data["budget"])
        name, banda = NOMBRES.get(opt, opt), tipo_banda(len(y))
        ax.plot(x, np.median(y, axis=0), drawstyle="steps-post", lw=2, color=COLORES[opt], label=f"{name}: mediana")
        if banda:
            lo, hi = np.percentile(y, [25, 75], axis=0) if banda == "p25-p75" else (y.min(axis=0), y.max(axis=0))
            ax.fill_between(x, lo, hi, step="post", color=COLORES[opt], alpha=0.2, lw=0,
                            label=f"{name}: {'percentiles 25–75' if banda == 'p25-p75' else 'mín–máx'}")
    # la mejora se concentra en las primeras evaluaciones: en escala lineal quedaría pegada al eje y
    ax.set_xscale("log")
    ax.set_xlabel("Evaluaciones de ruta completas (escala log)")
    ax.set_ylabel("Mejor costo encontrado (€)")
    ax.legend(fontsize=9)
    estilo(ax)
    titular(fig, f"Convergencia ACO vs GA, v = {v:g} €/h | budget {data['budget']:,}".replace(",", " ")
            + f"\n{nota_seeds(n)}", data)
    return fig


def figura_cajas(data, runs):
    values = sorted(runs["hourly_value_eur"].unique())
    n, cols = int(runs.groupby(["optimizer", "hourly_value_eur"]).size().min()), min(5, len(values))
    rows = -(-len(values) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.6 * rows), layout="constrained", squeeze=False)
    opts = list(dict.fromkeys(runs["optimizer"]))
    for ax, v in zip(axes.flat, values):
        for i, opt in enumerate(opts):
            costs = runs.loc[(runs["hourly_value_eur"] == v) & (runs["optimizer"] == opt), "best_cost"].to_numpy()
            if len(costs) >= MIN_SEEDS:
                ax.boxplot(costs, positions=[i], widths=0.5, showfliers=False, medianprops={"color": TEXTO},
                           boxprops={"color": GRIS}, whiskerprops={"color": GRIS}, capprops={"color": GRIS})
            # desplazamiento fijo (no aleatorio) para que la figura sea idéntica al regenerarla
            ax.scatter(i + np.linspace(-0.12, 0.12, len(costs)), costs, s=22, color=COLORES[opt],
                       edgecolor="white", lw=0.8, zorder=3, label=NOMBRES.get(opt, opt))
        ax.set_xticks(range(len(opts)), [NOMBRES.get(o, o) for o in opts])
        ax.set_title(f"v = {v:g} €/h", fontsize=10)
        estilo(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Costo final (€)")
    for ax in axes.flat[len(values):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower right", fontsize=9)
    fig.supxlabel("Cada panel tiene su propio eje y: compare ACO y GA dentro de un panel, no entre paneles.",
                  fontsize=9, color=GRIS, style="italic")
    titular(fig, f"Costo final por seed, ACO vs GA | {nota_seeds(n)}"
            + ("" if n < MIN_SEEDS else "\ncaja: cuartiles y mediana; bigotes: 1.5 RIC"), data)
    return fig


def figura_costo_v(data, summary):
    n = int(summary["n_runs"].min())
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    span = summary["hourly_value_eur"].max() - summary["hourly_value_eur"].min()
    for k, (opt, g) in enumerate(summary.groupby("optimizer", sort=False)):
        # pequeño desplazamiento horizontal: las medias de ACO y GA casi coinciden y se taparían
        ax.errorbar(g["hourly_value_eur"] + (k - 0.5) * 0.01 * span, g["mean_cost"],
                    yerr=g["std_cost"] if n >= 2 else None, marker="o", ms=5, lw=2, capsize=3,
                    color=COLORES[opt], label=f"{NOMBRES.get(opt, opt)}: media" + (" ± desv. estándar" if n >= 2 else ""))
    ax.set_xlabel("Valor de la hora v (€/h)")
    ax.set_ylabel("Costo total medio (€)")
    ax.legend(fontsize=9)
    estilo(ax)
    titular(fig, f"Costo total en función del valor de la hora\n{nota_seeds(n)}", data)
    return fig


def figura_desglose(data, summary):
    n = int(summary["n_runs"].min())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    panels = (("mean_time_h", "Horas de viaje (h)"), ("mean_toll_eur", "Peajes (€)"),
              ("mean_fuel_eur", "Combustible (€)"))
    span = summary["hourly_value_eur"].max() - summary["hourly_value_eur"].min()
    for ax, (col, label) in zip(axes, panels):
        for k, (opt, g) in enumerate(summary.groupby("optimizer", sort=False)):
            ax.errorbar(g["hourly_value_eur"] + (k - 0.5) * 0.01 * span, g[col],
                        yerr=g[col.replace("mean_", "std_")] if n >= 2 else None, marker="o", ms=5, lw=2,
                        capsize=3, color=COLORES[opt], label=NOMBRES.get(opt, opt))
        ax.set_xlabel("Valor de la hora v (€/h)")
        ax.set_ylabel(label)
        estilo(ax)
    axes[0].legend(fontsize=9)
    titular(fig, "Desglose de la ruta encontrada según v (media ± desv. estándar; las diferencias pequeñas pueden "
            f"ser variación entre seeds)\n{nota_seeds(n)}", data)
    return fig


def diferencia_medias(summary, a="aco", b="genetic"):
    """Media de b − media de a por v, con el error estándar de una diferencia de medias independientes."""
    s = summary.set_index(["hourly_value_eur", "optimizer"])
    A, B = s.xs(a, level="optimizer"), s.xs(b, level="optimizer")
    # las seeds de ACO y GA no comparten aleatoriedad: no se emparejan, se suman varianzas (Welch)
    se = np.sqrt(A["std_cost"] ** 2 / A["n_runs"] + B["std_cost"] ** 2 / B["n_runs"])
    return pd.DataFrame({"diff": B["mean_cost"] - A["mean_cost"], "se": se}).reset_index()


def figura_diferencia(data, summary):
    n, d = int(summary["n_runs"].min()), diferencia_medias(summary)
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.axhline(0, color=GRIS, lw=1)
    ax.errorbar(d["hourly_value_eur"], d["diff"], yerr=d["se"] if n >= 2 else None, marker="o", ms=6, lw=2,
                capsize=4, color=TEXTO)
    ax.set_xlabel("Valor de la hora v (€/h)")
    ax.set_ylabel("Costo medio GA − costo medio ACO (€)")
    estilo(ax)
    titular(fig, "Diferencia de costo medio GA − ACO según v (debajo de 0: GA mejor)\n"
            f"barras: ± 1 error estándar de la diferencia de medias (no emparejada) | {nota_seeds(n)}", data)
    return fig


def guardar(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Figuras del benchmark: convergencia, cajas, costo y desglose.")
    parser.add_argument("--benchmark", required=True, help="benchmark_runs.json")
    parser.add_argument("--summary", default="results/summary/benchmark_summary.csv")
    parser.add_argument("--hourly_value", type=float, help="v de la convergencia (por defecto, el primero)")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args(argv)

    data, runs, summary = cargar_benchmark(ROOT / args.benchmark, ROOT / args.summary)
    v = data["hourly_values_eur"][0] if args.hourly_value is None else args.hourly_value
    out = ROOT / args.out
    paths = [guardar(figura_convergencia(data, runs, v), out / f"convergence_v{v:g}.png"),
             guardar(figura_cajas(data, runs), out / "final_cost_boxplot.png"),
             guardar(figura_costo_v(data, summary), out / "cost_vs_v.png"),
             guardar(figura_desglose(data, summary), out / "cost_breakdown_vs_v.png")]
    if {"aco", "genetic"} <= set(summary["optimizer"]):  # la diferencia necesita ambos algoritmos
        paths.append(guardar(figura_diferencia(data, summary), out / "cost_diff_vs_v.png"))
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
