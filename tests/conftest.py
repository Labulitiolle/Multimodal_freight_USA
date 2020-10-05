import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point


@pytest.fixture()
def gen_nodes_edges_before_reindexing():
    nodes = gpd.GeoDataFrame(
        {
            "key": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "x": {1: 1.01, 2: 2.01, 3: -3.01, 4: -4.01, 5: -5.01},
            "y": {1: 1.02, 2: 1.02, 3: -1.02, 4: -4.01, 5: -5.01},
            "new_idx": {
                1: 1000000000,
                2: 1000000001,
                3: 1000000002,
                4: 1000000003,
                5: 1000000004,
            },
            "geometry": {
                1: Point(1.01, 1.02),
                2: Point(2.01, 2.02),
                3: Point(-3.01, -3.02),
                4: Point(-4.01, -4.01),
                5: Point(-5.01, -5.01),
            },
        },
        crs="EPSG:4326",
    )
    edges = gpd.GeoDataFrame(
        {
            "geometry": {
                0: LineString(),
                1: LineString(),
                2: LineString(),
            },
            "key": {0: 0, 1: 0, 2: 0},
            "u": {
                0: 1,
                1: 2,
                2: 4,
            },
            "v": {
                0: 2,
                1: 3,
                2: 5,
            },
        },
        crs="EPSG:4326",
    )
    return nodes, edges


@pytest.fixture()
def gen_nodes_edges_after_reindexing_two_components():
    nodes = gpd.GeoDataFrame(
        {
            "key": {
                1000000000: 0,
                1000000001: 0,
                1000000002: 0,
                1000000003: 0,
                1000000004: 0,
            },
            "x": {
                1000000000: 1.01,
                1000000001: 2.01,
                1000000002: -3.01,
                1000000003: -4.01,
                1000000004: -5.01,
            },
            "y": {
                1000000000: 1.02,
                1000000001: 1.02,
                1000000002: -1.02,
                1000000003: -4.01,
                1000000004: -5.01,
            },
            "geometry": {
                1000000000: Point(1.01, 1.02),
                1000000001: Point(2.01, 2.02),
                1000000002: Point(-3.01, -3.02),
                1000000003: Point(-4.01, -4.01),
                1000000004: Point(-5.01, -5.01),
            },
        },
        crs="EPSG:4326",
    )
    edges = gpd.GeoDataFrame(
        {
            "key": {0: 0, 1: 0, 2: 0},
            "u": {
                0: 1000000000,
                1: 1000000001,
                2: 1000000003,
            },
            "v": {
                0: 1000000001,
                1: 1000000002,
                2: 1000000004,
            },
        },
        crs="EPSG:4326",
    )
    return nodes, edges


@pytest.fixture()
def gen_nodes_edges_after_reindexing():
    nodes = gpd.GeoDataFrame(
        {
            "key": {1000000000: 0, 1000000001: 0, 1000000002: 0},
            "x": {1000000000: 1.01, 1000000001: 2.01, 1000000002: -3.01},
            "y": {1000000000: 1.02, 1000000001: 1.02, 1000000002: -1.02},
            "geometry": {
                1000000000: Point(1.01, 1.02),
                1000000001: Point(2.01, 2.02),
                1000000002: Point(-3.01, -3.02),
            },
        },
        crs="EPSG:4326",
    )
    edges = gpd.GeoDataFrame(
        {
            "length": {0: 1, 1: 2.1},
            "CO2_eq_kg": {0: 4, 1: 10},
            "duration_h": {0: 1, 1: 1},
            "key": {0: 0, 1: 0},
            "u": {0: 1000000000, 1: 1000000001},
            "v": {0: 1000000001, 1: 1000000002},
        },
        crs="EPSG:4326",
    )
    return nodes, edges
