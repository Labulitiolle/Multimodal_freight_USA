import pytest
import pandas as pd
import numpy as np

from mfreight.Multimodal.graph_utils import MultimodalNet


def test_get_rail_owners(mocker):
    mocker.patch(
        "osmnx.graph_to_gdfs",
        return_value=pd.DataFrame(
            {
                "RROWNER1": ["CSXT", "AGR", ["CSXT", "AGR"]],
                "RROWNER2": ["GFRR", ["GFRR", "CSXT"], "CSXT"],
            }
        ),
    )
    rail_owners = MultimodalNet().get_rail_owners()
    assert list(rail_owners) == ["CSXT", "AGR", "GFRR"]


def test_chose_operator(mocker):
    mocker.patch(
        "osmnx.graph_to_gdfs",
        return_value=(None, pd.DataFrame(
            {
                "trans_mode": ["rail", "rail", "rail", "rail", "raod"],
                "RROWNER1": ["CSXT", "AGR", ["CSXT", "AGR"], "BAYL", np.nan],
                "RROWNER2": ["GFRR", ["GFRR", "CSXT"], "CSXT", "BAYL", np.nan],
            }
        )),
    )
    mocker.patch("osmnx.graph_from_gdfs", return_value=None)

    edges = MultimodalNet().chose_operator(["AGR", "GFRR"])

def test_feature_scaling():
    x = np.array([0,5,1,10,2,0.1])

    result = MultimodalNet().feature_scaling(x)
