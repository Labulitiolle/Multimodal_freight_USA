import geopandas as gpd
import osmnx as ox
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from mfreight.Road.gen_road_net import RoadNet


def test_get_speed_data(mocker):
    mocker.patch(
        "mfreight.Road.gen_road_net.pd.read_html",
        return_value=[
            0,
            pd.DataFrame(
                {
                    "State or territory": {"0": "Alabama[9][10]", "1": "Alaska"},
                    "Freeway (rural)": {"0": "70mph (113km/h)", "1": "65mph (105km/h)"},
                    "Freeway (trucks)": {
                        "0": "70mph (113km/h)",
                        "1": "65mph (105km/h)",
                    },
                    "Freeway (urban)": {
                        "0": "55ph (89km/h)",
                        "1": "55mph (89km/h)",
                    },
                    "Divided (rural)": {"0": "65mph (105km/h)", "1": "55mph (89km/h)"},
                    "Undivided (rural)": {
                        "0": "55mph (89km/h)",
                        "1": "45mph (72km/h)",
                    },
                    "Residential": {"0": "20mph (32km/h)", "1": "20mph (32km/h)"},
                }
            ),
        ],
    )
    speed_table = RoadNet().get_speed_data()
    assert list(speed_table["State or territory"]) == ["Alabama", "Alaska"]


def test_map_state_to_STFIPS(mocker):
    mocker.patch(
        "mfreight.Road.gen_road_net.pd.read_html",
        return_value=[
            pd.DataFrame(
                {
                    "State Name": {
                        "0": "Alabama",
                        "1": "Alaska",
                        "2": "Arizona",
                        "3": "Occupation Information Industry",
                    },
                    "STFIPS code": {"0": 1.0, "1": 2.0, "2": 4.0, "3": None},
                    "Unnamed: 2": {
                        "0": None,
                        "1": None,
                        "2": None,
                        "3": "Printer-Friendly Version ",
                    },
                }
            ),
            2,
            3,
            4,
        ],
    )
    state_to_STFIPS_map = RoadNet().map_state_to_STFIPS()
    assert list(state_to_STFIPS_map.index) == ["Alabama", "Alaska", "Arizona"]
    assert list(state_to_STFIPS_map.values) == [1.0, 2.0, 4.0]


def test_STFIPS_to_speed_map(mocker):
    mocker.patch(
        "mfreight.Road.gen_road_net.RoadNet.get_speed_data",
        return_value=pd.DataFrame(
            {
                "State or territory": {"0": "Alabama", "1": "Alaska"},
                "Freeway (rural)": {"0": "70mph (113km/h)", "1": "65mph (105km/h)"},
                "Freeway (trucks)": {"0": "70mph (113km/h)", "1": "65mph (105km/h)"},
                "Freeway (urban)": {
                    "0": "55ph (89km/h)",
                    "1": "55mph (89km/h)",
                },
                "Divided (rural)": {"0": "65mph (105km/h)", "1": "55mph (89km/h)"},
                "Undivided (rural)": {
                    "0": "55mph (89km/h)",
                    "1": "45mph (72km/h)",
                },
                "Residential": {"0": "20mph (32km/h)", "1": "20mph (32km/h)"},
            }
        ),
    )
    mocker.patch(
        "mfreight.Road.gen_road_net.RoadNet.map_state_to_STFIPS",
        return_value=pd.Series({"Alabama": "0", "Alaska": "1", "Arizona": "2"}),
    )

    speed_map_kmh = RoadNet().STFIPS_to_speed_map()

    assert list(speed_map_kmh) == [113.0, 105.0]
    assert list(speed_map_kmh.index) == [0, 1]


def test_add_highway_speed(mocker):
    mocker.patch(
        "mfreight.Road.gen_road_net.RoadNet.STFIPS_to_speed_map",
        return_value=pd.Series({0: 70, 1: 100, 2: 80, 3: 80}),
    )

    edges = pd.DataFrame({"STFIPS": ["3", "1", "2"]})
    RoadNet().add_highway_speed(edges)

    assert list(edges.speed_kmh) == [80, 100, 80]


def test_add_incident_nodes():
    edges = pd.DataFrame(
        {
            "geometry": [
                LineString([(1.1, 2.1), (3.213, 34.33), (22.11, 1), (-12.3, 22)]),
                LineString([(1, 2), (-3.213, 2.33), (-2.3, 1)]),
            ]
        }
    )

    RoadNet().add_incident_nodes(edges)
    assert list(edges.u) == ["(1.1, 2.1)", "(1.0, 2.0)"]
    assert list(edges.v) == ["(-12.3, 22.0)", "(-2.3, 1.0)"]


def test_remove_attribute(gen_nodes_edges_before_reindexing):
    nodes, edges = gen_nodes_edges_before_reindexing

    graph = ox.graph_from_gdfs(nodes, edges)
    net = RoadNet(graph=graph)
    net.relabel_nodes(nodes)
    nodes, edges = ox.graph_to_gdfs(net.G)
    assert list(nodes.columns) == ["key", "x", "y", "geometry"]
    assert list(nodes.index) == [
        1000000000,
        1000000001,
        1000000002,
        1000000003,
        1000000004,
    ]
    assert list(edges.u) == [1000000000, 1000000001, 1000000003]


def test_simplify_graph(gen_nodes_edges_after_reindexing):
    nodes, edges = gen_nodes_edges_after_reindexing

    graph = ox.graph_from_gdfs(nodes, edges)
    net = RoadNet(graph=graph)
    net.simplify_graph()
    nodes, edges = ox.graph_to_gdfs(net.G)

    assert list(nodes.index) == [1000000000, 1000000002]
    assert list(edges["CO2_eq_kg"]) == [14]
    assert list(edges["length"]) == [3.1]
    assert list(edges["duration_h"]) == [2]


def test_keep_largest_component(gen_nodes_edges_after_reindexing_two_components):
    nodes, edges = gen_nodes_edges_after_reindexing_two_components

    graph = ox.graph_from_gdfs(nodes, edges)
    net = RoadNet(graph=graph.to_undirected())
    net.keep_largest_component()
    nodes, edges = ox.graph_to_gdfs(net.G)

    assert list(nodes.index) == [1000000000, 1000000001, 1000000002]


def test_gen_nodes_gdfs():
    edges = pd.DataFrame(
        {"u": ["(1.1, 2.1)", "(1.0, 2.0)"], "v": ["(-12.3, 22.0)", "(-2.3, 1.0)"]}
    )

    nodes = RoadNet().gen_nodes_gdfs(edges)

    result_df = pd.DataFrame(
        {
            "nodes_pos": ["(1.1, 2.1)", "(1.0, 2.0)", "(-12.3, 22.0)", "(-2.3, 1.0)"],
            "trans_mode": 4 * ["road"],
            "x": [1.1, 1.0, -12.3, -2.3],
            "y": [2.1, 2.0, 22.0, 1.0],
            "osmid": ["(1.1, 2.1)", "(1.0, 2.0)", "(-12.3, 22.0)", "(-2.3, 1.0)"],
            "geometry": [
                Point(1.1, 2.1),
                Point(1.0, 2.0),
                Point(-12.3, 22.0),
                Point(-2.3, 1.0),
            ],
            "new_idx": [1000000000, 1000000001, 1000000002, 1000000003],
        }
    )
    result_df.set_index("nodes_pos", drop=True, inplace=True)
    result = gpd.GeoDataFrame(result_df, crs="EPSG:4326")

    assert (nodes.x == result.x).all()
    assert (nodes.y == result.y).all()
    assert (nodes.osmid == result.osmid).all()
    assert (nodes.geometry == result.geometry).all()
    assert (nodes.new_idx == result.new_idx).all()


def test_gen_road_graph(mocker):
    mocker.patch(
        "mfreight.Road.gen_road_net.RoadNet.load_BTS",
        return_value=pd.DataFrame(
            {
                "OBJECTID": {"0": 1, "1": 2, "2": 3},
                "STATUS": {"0": 1, "1": 1, "2": 1},
                "BEGMP": {"0": 75.236, "1": 30.895, "2": 0.0},
                "ENDMP": {"0": 193.382, "1": 31.953, "2": 0.0},
                "KM": {"0": 29.081, "1": 0.689, "2": 3.869},
            }
        ),
    )
    mocker.patch(
        "mfreight.Road.gen_road_net.RoadNet.STFIPS_to_speed_map",
        return_value=pd.Series([80, 90, 100, 110]),
    )