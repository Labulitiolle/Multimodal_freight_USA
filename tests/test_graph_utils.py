import numpy as np
import geopandas as gpd
import pandas as pd
import pytest

from mfreight.Multimodal.graph_utils import MultimodalNet


def test_get_rail_owners(mocker):
    mocker.patch(
        "osmnx.graph_to_gdfs",
        return_value=(
            None,
            pd.DataFrame(
                {
                    "RROWNER1": ["CSXT", "AGR", ["CSXT", "AGR"]],
                    "RROWNER2": ["GFRR", ["GFRR", "CSXT"], "CSXT"],
                }
            ),
        ),
    )
    rail_owners = MultimodalNet().get_rail_owners()
    assert list(rail_owners) == ["CSXT", "AGR", "GFRR"]


def test_chose_operator(mocker):
    mocker.patch(
        "osmnx.graph_to_gdfs",
        return_value=(
            gpd.GeoDataFrame(
                {
                    "x": [1, 2, 3, 4, 5],
                    "y": [2, 3, 4, 5, 6],
                    "key": [0, 0, 0, 0, 0],
                    "geometry": [1,2,3,4,5],
                },
                crs="EPSG:4326",
            ),
            gpd.GeoDataFrame(
                {
                    "trans_mode": ["rail", "rail", "rail", "rail", "road"],
                    "RROWNER1": ["CSXT", "AGR", ["CSXT", "AGR"], "BAYL", np.nan],
                    "RROWNER2": ["GFRR", ["GFRR", "CSXT"], "CSXT", "BAYL", np.nan],
                    "u": [1, 2, 3, 4, 5],
                    "v": [2, 3, 4, 5, 6],
                    "key": [0, 0, 0, 0, 0],
                },
                crs="EPSG:4326",
            ),
        ),
    )
    mocker.patch("osmnx.graph_from_gdfs", return_value=None)

    edges = MultimodalNet().chose_operator(["AGR", "GFRR"])


def test_feature_scaling():
    x = np.array([0, 5, 1, 10])

    result = MultimodalNet().feature_scaling(x)

    assert list(result) == [0, 0.5, 0.1, 1]


def test_normalize_features(gen_edges_to_normalize):
    edges = gen_edges_to_normalize

    MultimodalNet().normalize_features(edges)

    assert list(edges.columns) == [
        "price",
        "duration_h",
        "CO2_eq_kg",
        "length",
        "price_normalized",
        "CO2_eq_kg_normalized",
        "duration_h_normalized",
    ]
    assert list(edges["price_normalized"]) == [0, 0.5, 0.1, 1]
    assert list(edges["CO2_eq_kg_normalized"]) == [0, 1, 0.1, 0.2]


testdata_target_feature = [
    (1, 0, 0, [0, 0.5, 0.1, 1]),
    (0, 0, 1, [0, 1, 0.1, 0.2]),
    (0.33, 0.33, 0.33, [0, 0.66, 0.1, 0.73]),
]


@pytest.mark.parametrize(
    "price_w, duration_w, CO2_eq_kg_w, expected", testdata_target_feature
)
def test_set_target_feature(
    price_w, duration_w, CO2_eq_kg_w, expected, gen_edges_normalized
):
    edges = gen_edges_normalized

    Net = MultimodalNet()
    Net.set_target_feature(edges, price_w, duration_w, CO2_eq_kg_w)

    assert list(round(edges.target_feature, 2)) == expected


testdata_set_price = [(["CA", "CO"], 2.153, 1.775), (["CA", "MD"], 0.837, 1.044)]


@pytest.mark.parametrize(
    "orig_dest, expected_intermodal_price, expected_truckload_price",
    testdata_set_price,
)
def test_get_price(
    orig_dest, expected_intermodal_price, expected_truckload_price, mocker
):
    mocker.patch(
        "mfreight.Multimodal.graph_utils.MultimodalNet.extract_state",
        side_effect=orig_dest,
    )
    intermodal_price, truckload_price = MultimodalNet().get_price((1, 2), (2, 3))

    assert intermodal_price == expected_intermodal_price
    assert truckload_price == expected_truckload_price


def test_set_price_to_edges(mocker):
    mocker.patch(
        "osmnx.graph_to_gdfs",
        return_value=(
            pd.DataFrame([]),
            pd.DataFrame(
                {
                    "trans_mode": ["rail", "road", "intermodal_link"],
                    "length": [1000, 1000, 1000],
                }
            ),
        ),
    )

    Net = MultimodalNet()
    Net.set_price_to_edges(intermodal_price=1, truckload_price=2)

    assert list(Net.edges.price) == [1, 2, 1.24]


def test_set_price_to_graph(gen_graph_for_price, mocker):
    graph = gen_graph_for_price
    mocker.patch(
        "networkx.read_gpickle",
        return_value=graph,
    )

    Net = MultimodalNet()
    Net.set_price_to_graph(intermodal_price=1, truckload_price=2)

    assert Net.G_multimodal_u[10000][10002][0]["price"] == 2
    assert Net.G_multimodal_u[10002][10003][0]["price"] == 1
    assert Net.G_multimodal_u[10003][10000][0]["price"] == 1.24


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
