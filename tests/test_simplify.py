import numpy as np
import geopandas as gpd
import osmnx as ox
import pytest
from mfreight.utils import simplify


def test_simpify():
    nodes = gpd.GeoDataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7],
            "y": [2, 3, 4, 5, 6, 7, 8],
            "key": [0, 0, 0, 0, 0, 0, 0],
            "geometry": [1, 2, 3, 4, 5, 6, 7],
        },
        crs="EPSG:4326",
    )
    edges = gpd.GeoDataFrame(
        {
            "trans_mode": ["rail", "rail", "rail", "rail", "rail", "rail"],
            "length": [1, 1, 1, 1, 1, 1],
            "RROWNER1": ["CSXT", "AGR", "CSXT", "CN", "CN", "NS"],
            "RROWNER2": ["GFRR", "CSXT", "BAYL", np.nan, np.nan, np.nan],
            "TRKRGHTS1": ["NS", "CSXT", np.nan, np.nan, np.nan, np.nan],
            "u": [0, 1, 2, 2, 4, 5],
            "v": [1, 2, 3, 4, 5, 6],
            "key": [0, 0, 0, 0, 0, 0],
        },
        crs="EPSG:4326",
    )

    G = ox.graph_from_gdfs(nodes, edges)

    G = simplify.simplify_graph(G)

    n, e = ox.graph_to_gdfs(G)


    assert len(G) == 4
    assert list(e["length"]) == [2, 1, 3]
    assert list(e["trans_mode"]) == ["rail", "rail", "rail"]
