"""Unit tests for Spanish mainland capitals road dataset (Omar)."""

import os
import numpy as np
import pandas as pd
import pytest

from src.data.cities import CANONICAL_CITIES, MANDATORY_EXCLUSIONS, get_cities_dataframe
from src.data.road_matrix import load_cities, load_distance_matrix, load_time_matrix, PROCESSED_DIR, DATA_DIR
from src.data.validate import validate_dataset


class TestCitiesContract:
    def test_canonical_cities_count(self):
        cities_df = get_cities_dataframe()
        assert len(cities_df) == 47

    def test_sequential_ids(self):
        cities_df = get_cities_dataframe()
        assert list(cities_df["id"]) == list(range(47))

    def test_unique_names_and_coordinates(self):
        cities_df = get_cities_dataframe()
        assert cities_df["name"].is_unique
        coords = list(zip(cities_df["lat"], cities_df["lon"]))
        assert len(set(coords)) == 47

    def test_mandatory_exclusions(self):
        cities_df = get_cities_dataframe()
        forbidden = set(cities_df["name"]).intersection(MANDATORY_EXCLUSIONS)
        assert len(forbidden) == 0, f"Ciudades prohibidas encontradas: {forbidden}"

    def test_peninsular_bounding_box(self):
        cities_df = get_cities_dataframe()
        # Península ibérica española: latitud 35.5N a 44N, longitud -10W a 4E
        assert cities_df["lat"].between(35.5, 44.0).all()
        assert cities_df["lon"].between(-10.0, 4.0).all()


class TestRoadMatrices:
    @pytest.fixture
    def matrices(self):
        D = load_distance_matrix(PROCESSED_DIR)
        T = load_time_matrix(PROCESSED_DIR)
        cities = load_cities(PROCESSED_DIR)
        return D, T, cities

    def test_matrices_shape(self, matrices):
        D, T, cities = matrices
        assert D.shape == (47, 47)
        assert T.shape == (47, 47)
        assert len(cities) == 47

    def test_diagonal_is_zero(self, matrices):
        D, T, _ = matrices
        np.testing.assert_array_equal(np.diag(D), np.zeros(47))
        np.testing.assert_array_equal(np.diag(T), np.zeros(47))

    def test_off_diagonal_strictly_positive_finite(self, matrices):
        D, T, _ = matrices
        mask = ~np.eye(47, dtype=bool)
        assert (D[mask] > 0).all()
        assert (T[mask] > 0).all()
        assert np.isfinite(D).all()
        assert np.isfinite(T).all()
        assert not np.isnan(D).any()
        assert not np.isnan(T).any()

    def test_asymmetry_property(self, matrices):
        D, T, _ = matrices
        # Carreteras reales y sentidos viales deben diferir
        assert not np.allclose(D, D.T, atol=1e-3)
        assert not np.allclose(T, T.T, atol=1e-3)

    def test_realistic_speeds(self, matrices):
        D, T, _ = matrices
        mask = ~np.eye(47, dtype=bool)
        speeds = D[mask] / T[mask]
        assert (speeds >= 40.0).all()
        assert (speeds <= 130.0).all()

    def test_known_highway_connections(self, matrices):
        D, T, cities = matrices
        name_to_id = {row["name"]: row["id"] for _, row in cities.iterrows()}
        madrid = name_to_id["Madrid"]
        bcn = name_to_id["Barcelona"]
        sevilla = name_to_id["Sevilla"]
        valencia = name_to_id["Valencia"]
        
        # Madrid - Barcelona: ~600-650 km
        assert 580.0 <= D[madrid, bcn] <= 660.0
        # Madrid - Valencia: ~330-380 km
        assert 330.0 <= D[madrid, valencia] <= 390.0
        # Madrid - Sevilla: ~500-560 km
        assert 500.0 <= D[madrid, sevilla] <= 570.0


class TestToyFixture:
    def test_toy_5cities_fixture(self):
        fixtures_dir = os.path.join(DATA_DIR, "fixtures")
        toy_csv = os.path.join(fixtures_dir, "toy_5cities.csv")
        toy_dist = os.path.join(fixtures_dir, "toy_5cities_distance_km.csv")
        toy_time = os.path.join(fixtures_dir, "toy_5cities_time_h.csv")
        
        assert os.path.exists(toy_csv)
        assert os.path.exists(toy_dist)
        assert os.path.exists(toy_time)
        
        df = pd.read_csv(toy_csv)
        assert len(df) == 5
        assert set(df["name"]) == {"Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza"}
        
        d_mat = pd.read_csv(toy_dist, index_col=0)
        t_mat = pd.read_csv(toy_time, index_col=0)
        assert d_mat.shape == (5, 5)
        assert t_mat.shape == (5, 5)
        assert np.all(np.diag(d_mat.values) == 0.0)
        assert np.all(np.diag(t_mat.values) == 0.0)


class TestFullValidator:
    def test_validate_dataset_function(self):
        report = validate_dataset(PROCESSED_DIR)
        assert report["status"] == "PASSED"
        assert report["cities_count"] == 47
        assert report["directed_pairs"] == 2162
