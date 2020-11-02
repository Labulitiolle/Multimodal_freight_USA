import numpy as np
import geopandas as gpd
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
    assert list(rail_owners) == ["CSXT", "NS"]


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
        "mfreight.Multimodal.graph_utils.MultimodalNet.chose_operator_in_df",
        return_value=(gen_edges_to_remove, None),
    )

    Net = MultimodalNet()
    MultimodalNet().chose_operator_in_graph(["CN", "NS"])

    assert len(Net.G_multimodal_u) == 7
    assert list(Net.G_multimodal_u.edges) == [(1, 2, 0), (4, 5, 0), (5, 6, 0)] #Road is not removed


def test_extract_state(mocker):
    mocker.patch(
        "geopy.geocoders.Nominatim",
        return_value="Florida, United States of America",  # TODO
    )

    state = Net.extract_state((30.439440, -85.057166))

    assert state == "FL"


def test_set_price_to_graph(mocker, gen_graph_for_operator_choice):
    mocker.patch(
        "MultimodalNet.G_multimodal_u",  # TODO
        return_value=gen_graph_for_operator_choice,
    )


# @pytest.mark.parametrize(
#     "orig_dest, expected_intermodal_price, expected_truckload_price",
#     testdata_set_price,
# )
# def test_get_price(
#     orig_dest, expected_intermodal_price, expected_truckload_price, mocker
# ):
#     mocker.patch(
#         "mfreight.Multimodal.graph_utils.MultimodalNet.extract_state",
#         side_effect=orig_dest,
#     )
#     intermodal_price, truckload_price = MultimodalNet().get_price((1, 2), (2, 3))
#
#     assert intermodal_price == expected_intermodal_price
#     assert truckload_price == expected_truckload_price


# def test_set_price_to_edges(mocker):
#     mocker.patch(
#         "osmnx.graph_to_gdfs",
#         return_value=(
#             pd.DataFrame([]),
#             pd.DataFrame(
#                 {
#                     "trans_mode": ["rail", "road", "intermodal_link"],
#                     "length": [1000, 1000, 1000],
#                 }
#             ),
#         ),
#     )
#
#     Net = MultimodalNet()
#     Net.set_price_to_edges(intermodal_price=1, truckload_price=2)
#
#     assert list(Net.edges.price) == [1, 2, 1.24]


def test_route_detail_from_graph(mocker, gen_graph_for_details):
    graph = gen_graph_for_details

    mocker.patch(
        "networkx.read_gpickle",
        return_value=graph,
    )

    Net = MultimodalNet()
    route_summary = Net.route_detail_from_graph(
        path=[10000, 10002, 10003, 10004],
        show_breakdown_by_mode=True,
        show_entire_route=False,
    )

    assert list(route_summary.index) == ["rail", "road", "Total"]
    assert list(route_summary.distance_km) == [3, 1, 4]
    assert list(route_summary.columns) == [
        "CO2_eq_kg",
        "duration_h",
        "distance_km",
        "price",
    ]
