import pytest
import pandas as pd
from Rail.gen_rail_net import RailNet



def test_keep_only_valid_usa_rail():
    nodes = pd.DataFrame({'FRANODEID': [1, 2, 3, 4, 5, 6]})
    edges = pd.DataFrame({'NET': ['I', 'M', 'O'], 'COUNTRY': ['US', 'MX', 'US'], 'u': [1, 2, 3], 'v': [2, 3, 4]})

    nodes, edges = RailNet().keep_only_valid_usa_rail(nodes,edges)
    assert (nodes.FRANODEID == [1,2]).all()
    assert (edges.NET == 'I').all()
