import pandas as pd
import pytest
from shapely.geometry import Point

from mfreight.Rail.gen_rail_net import RailNet


def test_keep_only_valid_usa_rail():
    edges = pd.DataFrame(
        {
            "NET": ["I", "M", "O"],
            "COUNTRY": ["US", "MX", "US"],
            "u": [1, 2, 3],
            "v": [2, 3, 4],
        }
    )

    RailNet().keep_only_valid_usa_rail(edges)
    assert list(edges.NET) == ["I"]


def test_add_speed_duration():
    edges = pd.DataFrame(
        {
            "TRACKS": [1, 2, 3, None],
            "MILES": [1.6, 4, 64, 1.6],
            "u": [1, 2, 3, 4],
            "v": [2, 3, 4, 5],
        }
    )
    RailNet().add_speed_duration(edges)
    assert list(edges.speed_mph) == [10, 25, 40, 10]
    assert list(edges.duration_h) == [0.16, 0.16, 1.6, 0.16]


def test_add_x_y_pos():
    nodes = pd.DataFrame(
        {
            "geometry": [
                Point(-123.3333, 45.3333),
                Point(-100.111, -12.3333),
                Point(150.3333, 99.2),
                Point(150.0, 99.00),
            ]
        }
    )
    RailNet().add_x_y_pos(nodes)
    assert list(nodes.x) == [-123.3333, -100.111, 150.3333, 150]
    assert list(nodes.y) == [45.3333, -12.3333, 99.2, 99]


def test_map_rail_to_intermodal_nodes():
    intermodal_nodes = pd.DataFrame(
        {
            "OBJECTID": [100, 200, 300, 400],
            "x": [-10.10, -10.111, 12.3333, 15.3333],
            "y": [-10.3333, 12.3333, 9.2, -8.2],
        }
    )
    nodes = pd.DataFrame(
        {
            "FRANODEID": [1, 2, 3, 4, 5],
            "x": [-19.9, -30.44, 90.33, -20, -100],
            "y": [-19.1, -40, 31.5, 15.4322, 17.12],
        }
    ).set_index("FRANODEID", drop=False)

    rail_node_map = RailNet().map_rail_to_intermodal_nodes(intermodal_nodes, nodes)

    assert (
        rail_node_map == pd.Series(index=[1, 4, 4, 1], data=[100, 200, 300, 400])
    ).all()


def test_add_intermodal_nodes(mocker):
    mocker.patch(
        "mfreight.Rail.gen_rail_net.RailNet.load_intermodal_facilities",
        return_value=pd.DataFrame(
            {
                "OBJECTID": [100, 200, 300, 400],
                "geometry": [
                    Point(-10.10, -10.3333),
                    Point(-10.111, 12.3333),
                    Point(12.3333, 9.2),
                    Point(15.3333, -8.2),
                ],
            }
        ),
    )
    nodes = pd.DataFrame(
        {
            "FRANODEID": [1, 2, 3, 4, 5],
            "x": [-19.9, -30.44, 90.33, -20, -100],
            "y": [-19.1, -40, 31.5, 15.4322, 17.12],
        }
    ).set_index("FRANODEID", drop=False)

    edges = pd.DataFrame(
        {
            "TRACKS": [1, 2, 3],
            "MILES": [1000, 1500, 2000],
            "u": [1, 2, 3],
            "v": [2, 3, 4],
        }
    )
    RailNet().add_intermodal_nodes(nodes, edges)
    assert list(nodes.FRANODEID) == [400, 2, 3, 300, 5]
    assert list(edges.u) == [400, 2, 3]


def test_extract_nested_operators():
    rail_edges = pd.DataFrame(
        {
            "RROWNER1": ["CSXT", "AGR", ["CSXT", "AGR"]],
            "RROWNER2": ["GFRR", ["GFRR", "CSXT"], "CSXT"],
            "TRKRGHTS2": ["GFRR", ["GFRR", "CSXT", "NS"], "CSXT"],
        }
    )
    RailNet().extract_nested_operators(rail_edges)
    print(rail_edges)
    assert list(rail_edges.iloc[0, :3]) == [
        "CSXT",
        "GFRR",
        "GFRR"
    ]
    assert list(rail_edges.iloc[1, :]) == [
        "AGR",
        "GFRR",
        "GFRR",
        "CSXT",
        "CSXT",
        "NS",
    ]
