"""Sensibilidad al valor de la hora y umbrales de cambio de ruta (Persona 4).

Para un ciclo fijo R:  J_R(v) = T_R * v + K_R,   T_R = tiempo total [h],  K_R = peajes + combustible [EUR].
Cruce entre A y B:     v* = (K_B - K_A) / (T_A - T_B)   (si T_A != T_B).

IMPORTANTE: los umbrales salen de las rutas CANDIDATAS que encontraron ACO/GA. Son valores
OBSERVADOS/aproximados; no prueban optimalidad (otra ruta no encontrada podría dominar).
"""
from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cost_matrix import CostComponents

__all__ = [
    "RouteLine",
    "Segment",
    "analyze_thresholds",
    "best_per_value",
    "candidate_lines",
    "canonical_route",
    "crossover",
    "line_for_route",
    "lower_envelope",
    "sweep",
    "thresholds",
    "write_sensitivity_csv",
]

OptimizeFn = Callable[[np.ndarray, int, int, Any], Mapping[str, Any]]
_TOL = 1e-9


def canonical_route(route: Sequence[int]) -> tuple[int, ...]:
    """Rota el ciclo para que empiece en el id menor (no cambia su costo)."""
    r = [int(x) for x in route]
    i = r.index(min(r))
    return tuple(r[i:] + r[:i])


@dataclass(frozen=True)
class RouteLine:
    route: tuple[int, ...]
    time_h: float
    fixed_eur: float

    def cost_at(self, hourly_value_eur: float) -> float:
        return self.time_h * hourly_value_eur + self.fixed_eur


@dataclass(frozen=True)
class Segment:
    v_from: float
    v_to: float
    line: RouteLine


def line_for_route(components: CostComponents, route: Sequence[int]) -> RouteLine:
    totals = components.route_totals(route)
    return RouteLine(canonical_route(route), totals.time_h, totals.fixed_eur)


def crossover(a: RouteLine, b: RouteLine) -> float | None:
    """v* donde J_A(v) = J_B(v); None si las pendientes (tiempos) coinciden."""
    dt = a.time_h - b.time_h
    if abs(dt) <= 1e-12:
        return None
    return (b.fixed_eur - a.fixed_eur) / dt


def lower_envelope(lines: Sequence[RouteLine], v_min: float, v_max: float) -> list[Segment]:
    """Ruta de menor costo entre las candidatas para cada v en [v_min, v_max]."""
    if not lines:
        raise ValueError("Se necesita al menos una ruta candidata")
    if v_min < 0 or v_max < v_min:
        raise ValueError("Se requiere 0 <= v_min <= v_max")

    costs = [ln.cost_at(v_min) for ln in lines]
    best = min(costs)
    tol = _TOL * max(1.0, abs(best))
    cur = min((ln for ln, c in zip(lines, costs) if c <= best + tol), key=lambda ln: ln.time_h)

    segments: list[Segment] = []
    v = float(v_min)
    while True:
        nxt, v_next = None, None
        for ln in lines:
            if ln.time_h >= cur.time_h - 1e-12:  # solo menor pendiente puede superar a cur al crecer v
                continue
            vc = (ln.fixed_eur - cur.fixed_eur) / (cur.time_h - ln.time_h)
            if vc <= v + _TOL or vc > v_max:
                continue
            if nxt is None or vc < v_next - _TOL or (abs(vc - v_next) <= _TOL and ln.time_h < nxt.time_h):
                nxt, v_next = ln, vc
        if nxt is None:
            segments.append(Segment(v, float(v_max), cur))
            return segments
        segments.append(Segment(v, v_next, cur))
        v, cur = v_next, nxt


def thresholds(segments: Sequence[Segment]) -> list[float]:
    """Valores de v en los que cambia la mejor ruta candidata."""
    return [s.v_from for s in segments[1:]]


def sweep(
    components: CostComponents,
    hourly_values: Sequence[float],
    optimize_fn: OptimizeFn,
    seeds: Sequence[int],
    budget: int,
    config: Any = None,
    optimizer: str = "optimizer",
) -> list[dict[str, Any]]:
    """Optimiza para cada (v, seed) y recalcula el costo por componentes."""
    rows: list[dict[str, Any]] = []
    for v in hourly_values:
        cm = components.cost_matrix(v)
        for seed in seeds:
            res = optimize_fn(cm, int(seed), int(budget), config)
            route = canonical_route(res["best_route"])
            totals = components.route_totals(route)
            recomputed = totals.cost(v)
            reported = float(res["best_cost"])
            if not np.isclose(recomputed, reported, rtol=1e-9, atol=1e-6):
                raise ValueError(
                    f"Costo reportado {reported} != recalculado {recomputed} "
                    f"(optimizer={optimizer}, v={v}, seed={seed})"
                )
            rows.append(
                {
                    "optimizer": optimizer,
                    "hourly_value_eur": float(v),
                    "seed": int(seed),
                    "best_cost": reported,
                    "time_h": totals.time_h,
                    "distance_km": totals.distance_km,
                    "toll_eur": totals.toll_eur,
                    "fuel_eur": totals.fuel_eur,
                    "evaluations": int(res.get("evaluations", -1)),
                    "tolls_complete": components.tolls_complete,
                    "route": route,
                }
            )
    return rows


def best_per_value(rows: Sequence[Mapping[str, Any]]) -> dict[float, Mapping[str, Any]]:
    """Mejor fila (menor costo) por valor horario, entre optimizadores y seeds."""
    best: dict[float, Mapping[str, Any]] = {}
    for row in rows:
        v = row["hourly_value_eur"]
        if v not in best or (row["best_cost"], row["route"]) < (best[v]["best_cost"], best[v]["route"]):
            best[v] = row
    return best


def candidate_lines(components: CostComponents, rows: Sequence[Mapping[str, Any]]) -> list[RouteLine]:
    """Rutas distintas (rotación canónica) halladas en el barrido."""
    unique = sorted({tuple(r["route"]) for r in rows})
    return [line_for_route(components, route) for route in unique]


def analyze_thresholds(
    components: CostComponents,
    rows: Sequence[Mapping[str, Any]],
    v_min: float,
    v_max: float,
) -> list[Segment]:
    return lower_envelope(candidate_lines(components, rows), v_min, v_max)


def write_sensitivity_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    if not rows:
        raise ValueError("No hay filas para escribir")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [k for k in rows[0] if k != "route"] + ["route"]
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "route": " ".join(str(c) for c in row["route"])})


def main(argv: Sequence[str] | None = None) -> None:
    """python -m src.costs.sensitivity --config config/sensitivity.yaml"""
    import argparse

    import yaml

    from ..optimizers.aco import ACO
    from ..optimizers.genetic import GeneticOptimizer
    from .cost_matrix import load_components
    from .vehicle import load_vehicle

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Barrido de valor horario (ACO y GA).")
    parser.add_argument("--config", default=str(root / "config" / "sensitivity.yaml"))
    parser.add_argument("--out", default=str(root / "results" / "sensitivity"))
    parser.add_argument("--toll-policy", choices=("error", "provisional_zero"), default=None)
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    vehicle = load_vehicle(cfg)
    policy = args.toll_policy or cfg.get("tolls", {}).get("policy", "error")
    components = load_components(vehicle, toll_policy=policy)

    sens = cfg["sensitivity"]
    registry = {
        "aco": (ACO, root / "config" / "aco.yaml"),
        "genetic": (GeneticOptimizer, root / "config" / "genetic.yaml"),
    }
    values = [float(v) for v in sens["hourly_values_eur"]]
    rows: list[dict[str, Any]] = []
    for name in sens["optimizers"]:
        cls, cfg_path = registry[name]
        with open(cfg_path, "r", encoding="utf-8") as handle:
            algo_cfg = yaml.safe_load(handle)
        rows += sweep(components, values, cls.optimize, sens["seeds"], sens["budget"], algo_cfg, name)

    out_dir = Path(args.out)
    write_sensitivity_csv(rows, out_dir / "sensitivity.csv")
    segments = analyze_thresholds(components, rows, min(values), max(values))
    summary = {
        "note": "Umbrales OBSERVADOS entre rutas candidatas de ACO/GA; no prueban optimalidad.",
        "tolls_complete": components.tolls_complete,
        "thresholds_eur_per_hour": thresholds(segments),
        "segments": [
            {"v_from": s.v_from, "v_to": s.v_to, "route": list(s.line.route),
             "time_h": s.line.time_h, "fixed_eur": s.line.fixed_eur}
            for s in segments
        ],
    }
    (out_dir / "thresholds.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Filas: {len(rows)} | umbrales observados (EUR/h): {summary['thresholds_eur_per_hour']}")


if __name__ == "__main__":
    main()
