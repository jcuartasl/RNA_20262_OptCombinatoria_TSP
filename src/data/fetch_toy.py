"""Fetch road distance and time from OSRM for the 5-city toy fixture."""

import json
import time
import requests
import numpy as np
import pandas as pd

TOY_CITIES = [
    {"id": 0, "name": "Madrid", "lat": 40.4168, "lon": -3.7038},
    {"id": 1, "name": "Barcelona", "lat": 41.3851, "lon": 2.1734},
    {"id": 2, "name": "Valencia", "lat": 39.4699, "lon": -0.3763},
    {"id": 3, "name": "Sevilla", "lat": 37.3891, "lon": -5.9845},
    {"id": 4, "name": "Zaragoza", "lat": 41.6488, "lon": -0.8891},
]


def fetch_osrm_table(cities):
    """Query OSRM table service for distance (m) and duration (s)."""
    # OSRM expects coordinates as {lon},{lat}
    coords_str = ";".join(f"{c['lon']},{c['lat']}" for c in cities)
    url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance,duration"
    
    headers = {"User-Agent": "UNAL_RNA_Academic_Project/1.0"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    data = response.json()
    
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM Error: {data}")
        
    distances_m = np.array(data["distances"], dtype=np.float64)
    durations_s = np.array(data["durations"], dtype=np.float64)
    
    # Convert meters -> km, seconds -> hours
    dist_km = distances_m / 1000.0
    time_h = durations_s / 3600.0
    
    # Force exact 0.0 on diagonal
    np.fill_diagonal(dist_km, 0.0)
    np.fill_diagonal(time_h, 0.0)
    
    return dist_km, time_h


if __name__ == "__main__":
    print("Testing OSRM connection with 5 cities...")
    dist_km, time_h = fetch_osrm_table(TOY_CITIES)
    names = [c["name"] for c in TOY_CITIES]
    
    print("\nMatriz de distancias (km):")
    df_dist = pd.DataFrame(dist_km, index=names, columns=names)
    print(df_dist.round(1))
    
    print("\nMatriz de tiempos (horas):")
    df_time = pd.DataFrame(time_h, index=names, columns=names)
    print(df_time.round(2))
    
    # Save toy fixture
    toy_df = pd.DataFrame(TOY_CITIES)
    toy_df["source"] = "IGN / OpenStreetMap"
    toy_path = r"..\staging_tsp_data\data\fixtures\toy_5cities.csv"
    toy_df.to_csv(toy_path, index=False, encoding="utf-8")
    
    # Also save toy distance and time matrices
    df_dist.to_csv(r"..\staging_tsp_data\data\fixtures\toy_5cities_distance_km.csv", encoding="utf-8")
    df_time.to_csv(r"..\staging_tsp_data\data\fixtures\toy_5cities_time_h.csv", encoding="utf-8")
    print(f"\nSaved toy fixtures to data/fixtures/")
