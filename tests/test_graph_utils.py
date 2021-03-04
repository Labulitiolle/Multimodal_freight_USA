import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from mfreight.Multimodal.graph_utils import MultimodalNet

Net = MultimodalNet()


def test_get_rail_owners(mocker):
    mocker.patch(
        "mfreight.Multimodal.graph_utils.MultimodalNet.load_and_format_csv",  # TODO
        return_value=gpd.GeoDataFrame(
            {
                "RROWNER1": ["CSXT", "AGR"],
                "RROWNER2": ["GFRR", "CSXT"],
                "TRKRGHTS1": ["NS", "CSXT"],
            }
        ),
    )
    rail_owners = MultimodalNet().get_rail_owners()
    assert "CSXT" in list(rail_owners)
    assert "NS" in list(rail_owners)


def test_chose_operator_in_df(mocker):
    mocker.patch(
        "mfreight.Multimodal.graph_utils.MultimodalNet.load_and_format_csv",  # TODO
        return_value=gpd.GeoDataFrame(
            {
                "trans_mode": ["rail", "rail", "rail", "rail"],
                "RROWNER1": ["CSXT", "AGR", "CSXT", "CN"],
                "RROWNER2": ["GFRR", "CSXT", "BAYL", np.nan],
                "TRKRGHTS1": ["NS", "CSXT", np.nan, np.nan],
                "u": [1, 2, 3, 4],
                "v": [2, 3, 4, 5],
                "key": [0, 0, 0, 0],
            },
            crs="EPSG:4326",
        ),
    )

    edges_to_remove, nodes_to_remove = MultimodalNet().chose_operator_in_df(
        ["CN", "NS"]
    )
    assert list(edges_to_remove.u) == [2, 3]
    assert list(edges_to_remove.v) == [3, 4]
    assert list(edges_to_remove.columns) == [
        "trans_mode",
        "RROWNER1",
        "RROWNER2",
        "TRKRGHTS1",
        "u",
        "v",
        "key",
    ]


def test_chose_operator_in_graph(
    mocker, gen_graph_for_operator_choice, gen_edges_to_remove
):
    mocker.patch(
        "networkx.read_gpickle",
        return_value=gen_graph_for_operator_choice,
    )
    mocker.patch(
        "mfreight.Multimodal.graph_utils.MultimodalNet.get_reachable_nodes",
        return_value=gen_graph_for_operator_choice,
    )
    mocker.patch(
        "mfreight.Multimodal.graph_utils.MultimodalNet.chose_operator_in_df",
        return_value=(gen_edges_to_remove, None),
    )

    Net = MultimodalNet()
    MultimodalNet().chose_operator_in_graph(
        G=gen_graph_for_operator_choice, operators=["CN", "NS"]
    )

    assert len(Net.G_multimodal_u) == 6
    assert list(Net.G_multimodal_u.edges) == [
        (1, 2),
        (4, 5),
        (5, 6),
    ]  # Road is not removed


def test_extract_state(mocker):
    mocker.patch(
        "geopy.geocoders.Nominatim",
        return_value="Florida, United States of America",
    )

    state = Net.extract_state((30.439440, -85.057166))

    assert state == "FL"


testdata = [("AR", "CA", "('AR', 'CA')"), ("FL", "FL", "range1")]


@pytest.mark.parametrize("state_1, state_2, expected_target_price", testdata)
def test_get_price_target(mocker, state_1, state_2, expected_target_price):
    mocker.patch(
        "mfreight.Multimodal.graph_utils.MultimodalNet.extract_state",
        side_effect=[state_1, state_2],
    )

    price_target = Net.get_price_target((1, 1), (2, 2))
    assert expected_target_price == price_target


def test_route_detail_from_graph(mocker, gen_graph_for_details):
    graph = gen_graph_for_details

    mocker.patch(
        "networkx.read_gpickle",
        return_value=graph,
    )

    Net = MultimodalNet()
    route_summary = Net.route_detail_from_graph(
        G=graph,
        path=[10000, 10002, 10003, 10004, 10005],
        show_breakdown_by_mode=True,
        show_entire_route=False,
    )

    assert list(route_summary.index) == ["intermodal_link", "rail", "Total"]
    assert list(route_summary.distance_miles) == [3000.0, 2000.0, 5000.0]
    assert list(route_summary.columns) == [
        "CO2_eq_kg",
        "distance_miles",
        "duration_h",
        "price",
    ]
