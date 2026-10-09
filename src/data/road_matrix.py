"""Acquisition and local management of road distance and time matrices for Spain's 47 mainland capitals."""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import requests

try:
    from .cities import CANONICAL_CITIES, get_cities_dataframe
except ImportError:
    from cities import CANONICAL_CITIES, get_cities_dataframe

# Directory paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_DIR = os.path.join(DATA_DIR, "raw")


def fetch_all_matrices() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Query OSRM table service for all 47 canonical cities and return distance and time matrices.
    
    Returns:
        cities_df: DataFrame of 47 cities
        dist_df: 47x47 DataFrame in km
        time_df: 47x47 DataFrame in hours
        metadata: Dict with metadata and manifest information
    """
    cities_df = get_cities_dataframe()
    cities_list = cities_df.to_dict(orient="records")
    n = len(cities_list)
    
    # OSRM expects {lon},{lat}
    coords_str = ";".join(f"{c['lon']},{c['lat']}" for c in cities_list)
    url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance,duration"
    
    query_timestamp = datetime.now(timezone.utc).isoformat()
    headers = {"User-Agent": "UNAL_RNA_Academic_Project/1.0"}
    
    print(f"Enviando consulta a OSRM para {n} ciudades ({n*(n-1)} pares dirigidos)...")
    response = requests.get(url, headers=headers, timeout=40)
    response.raise_for_status()
    payload = response.json()
    
    if payload.get("code") != "Ok":
        raise RuntimeError(f"OSRM Error: {payload}")
        
    distances_m = np.array(payload["distances"], dtype=np.float64)
    durations_s = np.array(payload["durations"], dtype=np.float64)
    
    # Convert meters -> km, seconds -> hours
    dist_km = distances_m / 1000.0
    time_h = durations_s / 3600.0
    
    # Enforce strict 0.0 on diagonal
    np.fill_diagonal(dist_km, 0.0)
    np.fill_diagonal(time_h, 0.0)
    
    # Check for NaN or negative values
    if np.isnan(dist_km).any() or np.isnan(time_h).any():
        raise ValueError("Se detectaron valores NaN en la respuesta de OSRM!")
    if (dist_km < 0).any() or (time_h < 0).any():
        raise ValueError("Se detectaron distancias o tiempos negativos!")
        
    city_names = cities_df["name"].tolist()
    dist_df = pd.DataFrame(dist_km, index=city_names, columns=city_names)
    time_df = pd.DataFrame(time_h, index=city_names, columns=city_names)
    
    # Compute integrity hash of matrices
    content_bytes = dist_df.to_csv().encode('utf-8') + time_df.to_csv().encode('utf-8')
    dataset_hash = hashlib.sha256(content_bytes).hexdigest()[:16]
    
    metadata = {
        "dataset_name": "47_mainland_spain_provincial_capitals_road_network",
        "dataset_version": "1.0.0",
        "dataset_hash_sha256_short": dataset_hash,
        "query_date_utc": query_timestamp,
        "source_provider": "Open Source Routing Machine (OSRM) Public API",
        "routing_profile": "driving (car)",
        "underlying_map_data": "OpenStreetMap contributors (ODbL)",
        "endpoint_url": "http://router.project-osrm.org/table/v1/driving",
        "dimension": n,
        "directed_pairs_count": n * (n - 1),
        "distance_unit": "kilometers (km)",
        "time_unit": "hours (h)",
        "symmetry": "asymmetric (real road network driving times and distances)",
        "diagonal": "0.0 (intra-city transition not valid)",
        "exclusions_verified": [
            "Palma (Illes Balears - insular)",
            "Las Palmas de Gran Canaria (Canarias - insular)",
            "Santa Cruz de Tenerife (Canarias - insular)",
            "Ceuta (Ciudad Autónoma - no peninsular)",
            "Melilla (Ciudad Autónoma - no peninsular)"
        ]
    }
    
    return cities_df, dist_df, time_df, metadata


def save_processed_dataset(cities_df: pd.DataFrame, dist_df: pd.DataFrame, time_df: pd.DataFrame, metadata: Dict[str, Any]):
    """Persist the dataset and metadata to data/processed and data/raw."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # Add source to cities_df
    cities_with_source = cities_df.copy()
    cities_with_source["source"] = "IGN / OpenStreetMap"
    
    cities_csv_path = os.path.join(PROCESSED_DIR, "cities.csv")
    dist_csv_path = os.path.join(PROCESSED_DIR, "distance_km.csv")
    time_csv_path = os.path.join(PROCESSED_DIR, "time_h.csv")
    meta_json_path = os.path.join(PROCESSED_DIR, "metadata.json")
    manifest_csv_path = os.path.join(RAW_DIR, "sources_manifest.csv")
    
    cities_with_source.to_csv(cities_csv_path, index=False, encoding="utf-8")
    dist_df.to_csv(dist_csv_path, encoding="utf-8")
    time_df.to_csv(time_csv_path, encoding="utf-8")
    
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    # Sources manifest
    manifest_df = pd.DataFrame([
        {
            "artifact": "cities.csv",
            "source": "Instituto Geográfico Nacional (IGN) / OpenStreetMap",
            "url": "https://www.ign.es / https://www.openstreetmap.org",
            "access_date": metadata["query_date_utc"],
            "license": "CC BY 4.0 / ODbL",
            "notes": "Coordenadas WGS84 de las 47 capitales provinciales peninsulares"
        },
        {
            "artifact": "distance_km.csv & time_h.csv",
            "source": "Open Source Routing Machine (OSRM) Table Service",
            "url": metadata["endpoint_url"],
            "access_date": metadata["query_date_utc"],
            "license": "ODbL (OSM data)",
            "notes": "Matriz asimétrica 47x47 de distancias por carretera y tiempos de conducción"
        }
    ])
    manifest_df.to_csv(manifest_csv_path, index=False, encoding="utf-8")
    print(f"Dataset guardado exitosamente en {PROCESSED_DIR} y {RAW_DIR}")


# Offline loaders for the team
def load_cities(data_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Load canonical cities table offline."""
    path = os.path.join(data_dir, "cities.csv")
    return pd.read_csv(path)


def load_distance_matrix(data_dir: str = PROCESSED_DIR) -> np.ndarray:
    """Load 47x47 distance matrix (km) as numpy float64 array offline."""
    path = os.path.join(data_dir, "distance_km.csv")
    df = pd.read_csv(path, index_col=0)
    return df.to_numpy(dtype=np.float64)


def load_time_matrix(data_dir: str = PROCESSED_DIR) -> np.ndarray:
    """Load 47x47 time matrix (hours) as numpy float64 array offline."""
    path = os.path.join(data_dir, "time_h.csv")
    df = pd.read_csv(path, index_col=0)
    return df.to_numpy(dtype=np.float64)
