"""Validation suite for the 47 mainland capitals road dataset."""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd
try:
    from .cities import MANDATORY_EXCLUSIONS
except ImportError:
    from cities import MANDATORY_EXCLUSIONS


def validate_dataset(data_dir: str) -> Dict[str, Any]:
    """Run comprehensive validations on cities.csv, distance_km.csv and time_h.csv.
    
    Raises:
        AssertionError or ValueError if any integrity rule is violated.
        
    Returns:
        Summary dict of validation metrics.
    """
    cities_path = os.path.join(data_dir, "cities.csv")
    dist_path = os.path.join(data_dir, "distance_km.csv")
    time_path = os.path.join(data_dir, "time_h.csv")
    
    for p in [cities_path, dist_path, time_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Archivo requerido no encontrado: {p}")
            
    # 1. Validar cities.csv
    cities_df = pd.read_csv(cities_path)
    assert len(cities_df) == 47, f"Esperadas exactamente 47 ciudades, encontradas {len(cities_df)}"
    assert list(cities_df["id"]) == list(range(47)), "Los IDs deben ser consecutivos de 0 a 46"
    assert cities_df["name"].is_unique, "Nombres de ciudades duplicados"
    
    # Exclusiones
    forbidden_found = set(cities_df["name"]).intersection(MANDATORY_EXCLUSIONS)
    assert len(forbidden_found) == 0, f"Se encontraron ciudades excluidas: {forbidden_found}"
    
    # Coordenadas en límites peninsulares de España
    assert (cities_df["lat"] >= 35.5).all() and (cities_df["lat"] <= 44.0).all(), "Latitudes fuera de la península ibérica"
    assert (cities_df["lon"] >= -10.0).all() and (cities_df["lon"] <= 4.0).all(), "Longitudes fuera de la península ibérica"
    
    # 2. Validar matrices
    dist_df = pd.read_csv(dist_path, index_col=0)
    time_df = pd.read_csv(time_path, index_col=0)
    
    assert dist_df.shape == (47, 47), f"Forma de distance_km debe ser 47x47, es {dist_df.shape}"
    assert time_df.shape == (47, 47), f"Forma de time_h debe ser 47x47, es {time_df.shape}"
    
    # Consistencia de etiquetas
    assert list(dist_df.index) == list(cities_df["name"]), "Índices de filas de distancias no coinciden con ciudades"
    assert list(dist_df.columns) == list(cities_df["name"]), "Columnas de distancias no coinciden con ciudades"
    assert list(time_df.index) == list(cities_df["name"]), "Índices de filas de tiempos no coinciden con ciudades"
    assert list(time_df.columns) == list(cities_df["name"]), "Columnas de tiempos no coinciden con ciudades"
    
    D = dist_df.to_numpy(dtype=np.float64)
    T = time_df.to_numpy(dtype=np.float64)
    
    # Ausencia de NaNs e infinitos
    assert not np.isnan(D).any(), "NaN detectado en matriz de distancias"
    assert not np.isnan(T).any(), "NaN detectado en matriz de tiempos"
    assert np.isfinite(D).all(), "Infinito detectado en matriz de distancias"
    assert np.isfinite(T).all(), "Infinito detectado en matriz de tiempos"
    
    # Diagonal nula
    assert np.all(np.diag(D) == 0.0), "La diagonal de distancias debe ser estrictamente 0.0"
    assert np.all(np.diag(T) == 0.0), "La diagonal de tiempos debe ser estrictamente 0.0"
    
    # Fuera de la diagonal estrictamente positivo
    mask_off_diag = ~np.eye(47, dtype=bool)
    assert (D[mask_off_diag] > 0).all(), "Distancias fuera de la diagonal deben ser estrictamente positivas"
    assert (T[mask_off_diag] > 0).all(), "Tiempos fuera de la diagonal deben ser estrictamente positivos"
    
    # Asimetría realista
    asym_dist = not np.allclose(D, D.T, atol=1e-3)
    asym_time = not np.allclose(T, T.T, atol=1e-3)
    
    # Velocidades medias razonables (km/h) para cada arco fuera de la diagonal
    speeds = D[mask_off_diag] / T[mask_off_diag]
    min_speed = float(np.min(speeds))
    max_speed = float(np.max(speeds))
    mean_speed = float(np.mean(speeds))
    assert min_speed >= 35.0, f"Velocidad mínima sospechosamente baja: {min_speed:.1f} km/h"
    assert max_speed <= 135.0, f"Velocidad máxima sospechosamente alta: {max_speed:.1f} km/h"
    
    return {
        "status": "PASSED",
        "cities_count": 47,
        "directed_pairs": 47 * 46,
        "min_distance_km": float(np.min(D[mask_off_diag])),
        "max_distance_km": float(np.max(D[mask_off_diag])),
        "mean_distance_km": float(np.mean(D[mask_off_diag])),
        "min_time_h": float(np.min(T[mask_off_diag])),
        "max_time_h": float(np.max(T[mask_off_diag])),
        "mean_time_h": float(np.mean(T[mask_off_diag])),
        "min_speed_km_h": min_speed,
        "max_speed_km_h": max_speed,
        "mean_speed_km_h": mean_speed,
        "distance_is_asymmetric": asym_dist,
        "time_is_asymmetric": asym_time,
    }


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "data", "processed"))
    report = validate_dataset(data_dir)
    print("=== REPORTE DE VALIDACIÓN DEL DATASET ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
