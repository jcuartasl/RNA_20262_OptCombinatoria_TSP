"""Canonical definition and validation of the 47 mainland Spanish provincial capitals."""

from typing import List, Dict, Any
import pandas as pd

# The 47 mainland provincial capitals of Spain in alphabetical order
# Coordenadas WGS84 (latitud, longitud) centradas en el núcleo urbano / red vial
CANONICAL_CITIES: List[Dict[str, Any]] = [
    {"id": 0, "name": "A Coruña", "province": "A Coruña", "community": "Galicia", "lat": 43.3623, "lon": -8.4115},
    {"id": 1, "name": "Albacete", "province": "Albacete", "community": "Castilla-La Mancha", "lat": 38.9943, "lon": -1.8585},
    {"id": 2, "name": "Alicante", "province": "Alicante", "community": "Comunitat Valenciana", "lat": 38.3452, "lon": -0.4810},
    {"id": 3, "name": "Almería", "province": "Almería", "community": "Andalucía", "lat": 36.8381, "lon": -2.4597},
    {"id": 4, "name": "Ávila", "province": "Ávila", "community": "Castilla y León", "lat": 40.6565, "lon": -4.6818},
    {"id": 5, "name": "Badajoz", "province": "Badajoz", "community": "Extremadura", "lat": 38.8794, "lon": -6.9707},
    {"id": 6, "name": "Barcelona", "province": "Barcelona", "community": "Cataluña", "lat": 41.3851, "lon": 2.1734},
    {"id": 7, "name": "Bilbao", "province": "Bizkaia", "community": "País Vasco", "lat": 43.2630, "lon": -2.9350},
    {"id": 8, "name": "Burgos", "province": "Burgos", "community": "Castilla y León", "lat": 42.3440, "lon": -3.6969},
    {"id": 9, "name": "Cáceres", "province": "Cáceres", "community": "Extremadura", "lat": 39.4753, "lon": -6.3723},
    {"id": 10, "name": "Cádiz", "province": "Cádiz", "community": "Andalucía", "lat": 36.5271, "lon": -6.2886},
    {"id": 11, "name": "Castellón de la Plana", "province": "Castellón", "community": "Comunitat Valenciana", "lat": 39.9864, "lon": -0.0513},
    {"id": 12, "name": "Ciudad Real", "province": "Ciudad Real", "community": "Castilla-La Mancha", "lat": 38.9861, "lon": -3.9274},
    {"id": 13, "name": "Córdoba", "province": "Córdoba", "community": "Andalucía", "lat": 37.8882, "lon": -4.7794},
    {"id": 14, "name": "Cuenca", "province": "Cuenca", "community": "Castilla-La Mancha", "lat": 40.0704, "lon": -2.1374},
    {"id": 15, "name": "Girona", "province": "Girona", "community": "Cataluña", "lat": 41.9794, "lon": 2.8214},
    {"id": 16, "name": "Granada", "province": "Granada", "community": "Andalucía", "lat": 37.1773, "lon": -3.5986},
    {"id": 17, "name": "Guadalajara", "province": "Guadalajara", "community": "Castilla-La Mancha", "lat": 40.6337, "lon": -3.1674},
    {"id": 18, "name": "Huelva", "province": "Huelva", "community": "Andalucía", "lat": 37.2614, "lon": -6.9447},
    {"id": 19, "name": "Huesca", "province": "Huesca", "community": "Aragón", "lat": 42.1362, "lon": -0.4087},
    {"id": 20, "name": "Jaén", "province": "Jaén", "community": "Andalucía", "lat": 37.7796, "lon": -3.7849},
    {"id": 21, "name": "León", "province": "León", "community": "Castilla y León", "lat": 42.5987, "lon": -5.5671},
    {"id": 22, "name": "Lleida", "province": "Lleida", "community": "Cataluña", "lat": 41.6176, "lon": 0.6200},
    {"id": 23, "name": "Logroño", "province": "La Rioja", "community": "La Rioja", "lat": 42.4658, "lon": -2.4499},
    {"id": 24, "name": "Lugo", "province": "Lugo", "community": "Galicia", "lat": 43.0125, "lon": -7.5558},
    {"id": 25, "name": "Madrid", "province": "Madrid", "community": "Comunidad de Madrid", "lat": 40.4168, "lon": -3.7038},
    {"id": 26, "name": "Málaga", "province": "Málaga", "community": "Andalucía", "lat": 36.7213, "lon": -4.4214},
    {"id": 27, "name": "Murcia", "province": "Murcia", "community": "Región de Murcia", "lat": 37.9922, "lon": -1.1307},
    {"id": 28, "name": "Ourense", "province": "Ourense", "community": "Galicia", "lat": 42.3358, "lon": -7.8639},
    {"id": 29, "name": "Oviedo", "province": "Asturias", "community": "Principado de Asturias", "lat": 43.3619, "lon": -5.8494},
    {"id": 30, "name": "Palencia", "province": "Palencia", "community": "Castilla y León", "lat": 42.0095, "lon": -4.5288},
    {"id": 31, "name": "Pamplona", "province": "Navarra", "community": "Comunidad Foral de Navarra", "lat": 42.8125, "lon": -1.6458},
    {"id": 32, "name": "Pontevedra", "province": "Pontevedra", "community": "Galicia", "lat": 42.4310, "lon": -8.6444},
    {"id": 33, "name": "Salamanca", "province": "Salamanca", "community": "Castilla y León", "lat": 40.9701, "lon": -5.6635},
    {"id": 34, "name": "San Sebastián", "province": "Gipuzkoa", "community": "País Vasco", "lat": 43.3183, "lon": -1.9812},
    {"id": 35, "name": "Santander", "province": "Cantabria", "community": "Cantabria", "lat": 43.4647, "lon": -3.8044},
    {"id": 36, "name": "Segovia", "province": "Segovia", "community": "Castilla y León", "lat": 40.9429, "lon": -4.1088},
    {"id": 37, "name": "Sevilla", "province": "Sevilla", "community": "Andalucía", "lat": 37.3891, "lon": -5.9845},
    {"id": 38, "name": "Soria", "province": "Soria", "community": "Castilla y León", "lat": 41.7640, "lon": -2.4688},
    {"id": 39, "name": "Tarragona", "province": "Tarragona", "community": "Cataluña", "lat": 41.1189, "lon": 1.2445},
    {"id": 40, "name": "Teruel", "province": "Teruel", "community": "Aragón", "lat": 40.3456, "lon": -1.1072},
    {"id": 41, "name": "Toledo", "province": "Toledo", "community": "Castilla-La Mancha", "lat": 39.8628, "lon": -4.0273},
    {"id": 42, "name": "Valencia", "province": "Valencia", "community": "Comunitat Valenciana", "lat": 39.4699, "lon": -0.3763},
    {"id": 43, "name": "Valladolid", "province": "Valladolid", "community": "Castilla y León", "lat": 41.6523, "lon": -4.7245},
    {"id": 44, "name": "Vitoria-Gasteiz", "province": "Álava", "community": "País Vasco", "lat": 42.8469, "lon": -2.6716},
    {"id": 45, "name": "Zamora", "province": "Zamora", "community": "Castilla y León", "lat": 41.5063, "lon": -5.7446},
    {"id": 46, "name": "Zaragoza", "province": "Zaragoza", "community": "Aragón", "lat": 41.6488, "lon": -0.8891},
]

MANDATORY_EXCLUSIONS = {
    "Palma", "Palma de Mallorca",
    "Las Palmas de Gran Canaria", "Las Palmas",
    "Santa Cruz de Tenerife",
    "Ceuta", "Melilla"
}


def get_cities_dataframe() -> pd.DataFrame:
    """Return canonical DataFrame of the 47 mainland capitals."""
    df = pd.DataFrame(CANONICAL_CITIES)
    # Validation checks
    assert len(df) == 47, f"Expected exactly 47 cities, got {len(df)}"
    assert df["id"].is_unique, "IDs must be unique"
    assert df["name"].is_unique, "City names must be unique"
    assert set(df["name"]).isdisjoint(MANDATORY_EXCLUSIONS), "Found forbidden insular/autonomous cities!"
    return df
