"""All credits to Geoff Boeing, this script was taken from his osmnx repository and adapted for my use."""

import networkx as nx
import pandas as pd
import time
from shapely.geometry import LineString


def graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=None, undirected=False):
    """
    Convert node and edge GeoDataFrames to a MultiDiGraph.
    This function is the inverse of `graph_to_gdfs`.
    Parameters
    ----------
    gdf_nodes : geopandas.GeoDataFrame
        GeoDataFrame of graph nodes
    gdf_edges : geopandas.GeoDataFrame
        GeoDataFrame of graph edges, must have crs attribute set
    graph_attrs : dict
        the new G.graph attribute dict; if None, add crs as the only
        graph-level attribute
    undirected : bool
        Duplicate each edge to make the graph undirected.
    Returns
    -------
    G : networkx.MultiDiGraph
    """
    if graph_attrs is None:
        graph_attrs = {"crs": gdf_edges.crs}
    G = nx.MultiDiGraph(**graph_attrs)

    # add the nodes then each node's non-null attributes
    start = time.time()
    G.add_nodes_from(gdf_nodes.index)
    for col in gdf_nodes.columns:
        nx.set_node_attributes(G, name=col, values=gdf_nodes[col].dropna())
    print(f"time elapsed: {time.time() - start}s")
    # add each edge and its non-null attributes
    if undirected:
        for (u, v, k), row in gdf_edges.set_index(["u", "v", "key"]).iterrows():
            d = {
                label: val
                for label, val in row.items()
                if isinstance(val, list) or pd.notnull(val)
            }
            d_rev = d.copy()
            d_rev["geometry"] = LineString(d["geometry"].coords[::-1])

            G.add_edge(u, v, k, **d)
            G.add_edge(v, u, k, **d_rev)
    else:
        start = time.time()
        for (u, v, k), row in gdf_edges.set_index(["u", "v", "key"]).iterrows():
            d = {
                label: val
                for label, val in row.items()
                if isinstance(val, list) or pd.notnull(val)
            }

            G.add_edge(u, v, k, **d)
        print(f"time elapsed: {time.time() - start}s")
    return G


def graph_from_gdfs_revisited(gdf_nodes, gdf_edges, graph_attrs=None):

    if graph_attrs is None:
        graph_attrs = {"crs": gdf_edges.crs}

    g = nx.MultiDiGraph(**graph_attrs)

    reserved_columns = ["u", "v", "key"]
    attr_col_headings = [c for c in gdf_edges.columns if c not in reserved_columns]
    attribute_data = zip(*[gdf_edges[col] for col in attr_col_headings])

    for s, t, k, attrs in zip(
        gdf_edges["u"], gdf_edges["v"], gdf_edges["key"], attribute_data
    ):

        key = g.add_edge(s, t, key=k)

        g[s][t][key].update(zip(attr_col_headings, attrs))

    for col in ["x", "y", "geometry"]:  # gdf_nodes.columns:
        nx.set_node_attributes(g, name=col, values=gdf_nodes[col].dropna())

    return g
