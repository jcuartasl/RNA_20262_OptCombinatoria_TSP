"""Data package for Spanish mainland capitals TSP."""

from .cities import CANONICAL_CITIES, MANDATORY_EXCLUSIONS, get_cities_dataframe
from .road_matrix import load_cities, load_distance_matrix, load_time_matrix
from .validate import validate_dataset

__all__ = [
    "CANONICAL_CITIES",
    "MANDATORY_EXCLUSIONS",
    "get_cities_dataframe",
    "load_cities",
    "load_distance_matrix",
    "load_time_matrix",
    "validate_dataset",
]
