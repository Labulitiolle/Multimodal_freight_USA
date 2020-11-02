import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
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
    )
    return nodes, edges


@pytest.fixture()
def gen_formatted_rail_and_road_nodes():
    road_nodes = gpd.GeoDataFrame(
        index=[10000, 10002, 10003],
        data={
            "trans_mode": ["road", "road", "road"],
            "x": [-22, 2, 1],
            "y": [-30.1, 3, 1],
            "osmid": [10000, 10002, 1003],
            "geometry": [Point(-22, -30.1), Point(2, 3), Point(1, 1)],
            "key": [0, 0, 0],
            "STCYFIPS": [12343, 12345, 12123],
        },
    )

    rail_nodes = gpd.GeoDataFrame(
        index=[30001, 2, 10],
        data={
            "FRANODEID": [30001, 2, 10],
            "STATE": ["FL", "FL", "FL"],
            "STCYFIPS": [12354, 12321, 12111],
            "geometry": [Point(3, 3), Point(1.9, 3), Point(0.5, 1)],
            "trans_mode": ["rail", "intermodal", "intermodal"],
            "key": [0, 0, 0],
            "x": [3, 1.9, 0.5],
            "y": [3, 3, 1],
        },
    )

    return road_nodes, rail_nodes


@pytest.fixture()
def gen_edges_to_normalize():
    return gpd.GeoDataFrame(
        {
            "price": [0, 5, 1, 10],
            "duration_h": [0, 50, 10, 100],
            "CO2_eq_kg": [1, 11, 2, 3],
            "length": [1, 2, 3, 4],
        }
    )


@pytest.fixture()
def gen_edges_normalized():
    return gpd.GeoDataFrame(
        {
            "price": [0, 5, 1, 10],
            "duration_h": [0, 50, 10, 100],
            "CO2_eq_kg": [1, 11, 2, 3],
            "length": [1, 2, 3, 4],
            "price_normalized": [0, 0.5, 0.1, 1],
            "duration_h_normalized": [0, 0.5, 0.1, 1],
            "CO2_eq_kg_normalized": [0, 1, 0.1, 0.2],
        }
    )


@pytest.fixture()
def gen_graph_for_price():
    nodes = gpd.GeoDataFrame(
        index=[10000, 10002, 10003],
        data={
            "trans_mode": ["road", "rail", "intermodal"],
            "x": [-22, 2, 1],
            "y": [-30.1, 3, 1],
            "osmid": [10000, 10002, 1003],
            "geometry": [Point(-22, -30.1), Point(2, 3), Point(1, 1)],
            "key": [0, 0, 0],
            "STCYFIPS": [12343, 12345, 12123],
        },
    )

    edges = gpd.GeoDataFrame(
        {
            "trans_mode": {0: "road", 1: "rail", 2: "intermodal_link"},
            "length": {0: 1000, 1: 1000, 2: 1000},
            "key": {0: 0, 1: 0, 2: 0},
            "u": {0: 10000, 1: 10002, 2: 10003},
            "v": {0: 10002, 1: 10003, 2: 10000},
        },
    )

    return ox.graph_from_gdfs(nodes, edges)


@pytest.fixture()
def gen_graph_for_details():
    nodes = gpd.GeoDataFrame(
        index=[10000, 10002, 10003, 10004],
        data={
            "trans_mode": ["road", "rail", "intermodal", "rail"],
            "x": [-22, 2, 1, 2],
            "y": [-30.1, 3, 1, 1],
            "osmid": [10000, 10002, 10003, 10004],
            "geometry": [Point(-22, -30.1), Point(2, 3), Point(1, 1), Point(2, 1)],
            "key": [0, 0, 0, 0],
            "STCYFIPS": [12343, 12345, 12123, 12124],
        },
    )

    edges = gpd.GeoDataFrame(
        {
            "trans_mode": {0: "road", 1: "rail", 2: "intermodal_link", 3: "rail"},
            "length": {0: 1000, 1: 1000, 2: 1000, 3: 2000},
            "key": {0: 0, 1: 0, 2: 0, 3: 0},
            "u": {0: 10000, 1: 10002, 2: 10003, 3: 10003},
            "v": {0: 10002, 1: 10003, 2: 10000, 3: 10004},
        }
    )

    return ox.graph_from_gdfs(nodes, edges)


@pytest.fixture()
def gen_graph_for_operator_choice():
    nodes = gpd.GeoDataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 3, 4, 5, 6],
            "key": [0, 0, 0, 0, 0],
            "geometry": [1, 2, 3, 4, 5],
        },
        crs="EPSG:4326",
    )
    edges = gpd.GeoDataFrame(
        {
            "trans_mode": ["rail", "rail", "rail", "rail", "road"],
            "RROWNER1": ["CSXT", "AGR", "CSXT", "CN", np.nan],
            "RROWNER2": ["GFRR", "CSXT", "BAYL", np.nan, np.nan],
            "TRKRGHTS1": ["NS", "CSXT", np.nan, np.nan, np.nan],
            "u": [1, 2, 3, 4, 5],
            "v": [2, 3, 4, 5, 6],
            "key": [0, 0, 0, 0, 0],
        },
        crs="EPSG:4326",
    )

    return ox.graph_from_gdfs(nodes, edges)


@pytest.fixture()
def gen_edges_to_remove():
    return gpd.GeoDataFrame(
        {
            "trans_mode": ["rail", "rail"],
            "RROWNER1": ["AGR", "CSXT"],
            "RROWNER2": ["CSXT", "BAYL"],
            "TRKRGHTS1": ["CSXT", np.nan],
            "u": [2, 3],
            "v": [3, 4],
            "key": [0, 0],
        },
        crs="EPSG:4326",
    )
