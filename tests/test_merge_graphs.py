import geopandas as gpd
import pytest
from shapely.geometry import LineString

from mfreight.Multimodal.merge_graphs import MergeNets, keep_only_intermodal


def test_keep_only_intermodal():
    nodes = gpd.GeoDataFrame(
        index=[3001, 88, 177, 3100, 3200],
        data={
            "trans_mode": ["rail", "intermodal", "intermodal", "rail", "rail"],
            "STCYFIPS": [12345, 23123, 43121, 12345, 19932],
        },
    )
    nodes = keep_only_intermodal(nodes)

    assert list(nodes.index) == [88, 177]
    assert list(nodes.trans_mode) == ["intermodal", "intermodal"]


def test_map_intermodal_to_road(mocker):
    mocker.patch(
        "mfreight.Multimodal.merge_graphs.keep_only_intermodal",
        return_value=gpd.GeoDataFrame(
            index=[2, 5, 10], data={"x": [1, 2, -3], "y": [1, 2, -3]}
        ),
    )

    road_nodes = gpd.GeoDataFrame(
        index=[1001, 1002, 1003, 1004, 1005],
        data={"x": [-2, -1.5, 2.5, 1, -0.2], "y": [-5, -2, 1.7, 1.5, 5]}
    )

    intermodal_to_road_map = MergeNets().map_intermodal_to_road(road_nodes, None)

    assert list(intermodal_to_road_map.index) == [2, 5, 10]
    assert list(intermodal_to_road_map.road_idx) == [1004, 1003, 1002]
    assert list(intermodal_to_road_map.dist) == [34.55, 40.27, 124.48]


def test_gen_intermodal_links():
    intermodal_to_road_map = gpd.GeoDataFrame(
        index=[2, 5, 10],
        data={
            "road_idx": [1004, 1003, 1002],
            "dist": [100, 200, 10],
        },
    )

    road_edges = gpd.GeoDataFrame(
        {
            "STATUS": [0],
            "trans_mode": ["road"],
            "MILES": [1000],
            "duration_h": [0.1],
            "CO2_eq_kg": [1],
            "geometry": [LineString()],
            "u": [1],
            "v": [2],
            "key": [0],
        }
    )

    intermodal_links = MergeNets(kg_co2_per_tmiles=10).gen_intermodal_links(
        intermodal_to_road_map, road_edges
    )

    assert list(intermodal_links.u) == [1004, 1003, 1002]
    assert list(intermodal_links.v) == [2, 5, 10]
    assert list(intermodal_links.CO2_eq_kg) == [1000, 2000, 100]


def test_gen_intermodal_nodes(gen_formatted_rail_and_road_nodes):
    intermodal_to_road_map = gpd.GeoDataFrame(
        index=[2, 10],
        data={"road_idx": [1004, 1002], "dist": [55589.07, 200410.48]},
    )

    road_nodes, rail_nodes = gen_formatted_rail_and_road_nodes
    intermodal_nodes = MergeNets().gen_intermodal_nodes(
        intermodal_to_road_map, road_nodes, rail_nodes
    )

    assert list(intermodal_nodes.index) == [2, 10]
    assert list(intermodal_nodes.x) == [1.9, 0.5]
    assert list(intermodal_nodes.columns) == [
        "trans_mode",
        "x",
        "y",
        "osmid",
        "geometry",
        "key",
        "STCYFIPS",
    ]


def test_link_road_to_rail(gen_formatted_rail_and_road_nodes):

    road_edges = gpd.GeoDataFrame(
        index=[101, 102, 103],
        data={
            "STATUS": [1, 1, 1],
            "trans_mode": ["road", "road", "road"],
            "MILES": [1000, 10000, 20000],
            "duration_h": [0.1, 1, 2],
            "CO2_eq_kg": [0.01, 0.1, 0.2],
            "geometry": [LineString(), LineString(), LineString()],
            "u": [10000, 10002, 10003],
            "v": [10002, 10003, 10000],
            "key": [0, 0, 0],
        },
    )

    road_nodes, rail_nodes = gen_formatted_rail_and_road_nodes

    road_nodes, road_edges = MergeNets().link_road_to_rail(
        road_nodes, rail_nodes, road_edges
    )

    assert list(road_nodes.index) == [10000, 10002, 10003, 2, 10]
    assert list(road_nodes.trans_mode) == [
        "road",
        "road",
        "road",
        "intermodal",
        "intermodal",
    ]
    assert list(road_edges.index) == [0, 1, 2, 3, 4]
    assert list(road_edges.u) == [10000, 10002, 10003, 10002, 10003]
    assert list(road_edges.v) == [10002, 10003, 10000, 2, 10]
    assert list(road_edges.duration_h) == [0.1, 1, 2, 2, 2]
