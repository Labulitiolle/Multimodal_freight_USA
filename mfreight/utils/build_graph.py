"""All credits to Geoff Boeing, this script was taken from his osmnx repository and adapted for my use."""

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from osmnx import utils
from shapely.geometry import LineString, Point


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
    G.add_nodes_from(gdf_nodes.index)
    for col in gdf_nodes.columns:
        nx.set_node_attributes(G, name=col, values=gdf_nodes[col].dropna())
    # add each edge and its non-null attributes
    if undirected:
        for (u, v, k), row in gdf_edges.set_index(["u", "v", "key"]).iterrows():
            d = {
                label: val
                for label, val in row.items()
                if isinstance(val, list) or pd.notnull(val)
            }
            G.add_edge(u, v, k, **d)

            if "geometry" in d.keys():
                d_rev = d.copy()
                d_rev["geometry"] = LineString(d["geometry"].coords[::-1])
                G.add_edge(v, u, k, **d_rev)
            else:
                G.add_edge(v, u, k, **d)
    else:
        for (u, v, k), row in gdf_edges.set_index(["u", "v", "key"]).iterrows():
            d = {
                label: val
                for label, val in row.items()
                if isinstance(val, list) or pd.notnull(val)
            }

            G.add_edge(u, v, k, **d)
    return G


def graph_from_gdfs2(gdf_nodes, gdf_edges, graph_attrs=None):
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

    Returns
    -------
    G : networkx.MultiDiGraph
    """
    if graph_attrs is None:
        graph_attrs = {"crs": gdf_edges.crs}
    G = nx.Graph(**graph_attrs)

    # assemble edges' attribute names and values
    attr_names = [c for c in gdf_edges.columns if c not in {"u", "v"}]
    attr_values = zip(*[gdf_edges[col] for col in attr_names])

    # add edges and their attributes to graph, but filter out null attribute
    # values so that edges only get attributes with non-null values
    for u, v, edge_vals in zip(gdf_edges["u"], gdf_edges["v"], attr_values):
        edge_attrs = zip(attr_names, edge_vals)
        data = {
            name: val
            for name, val in edge_attrs
            if isinstance(val, list) or pd.notnull(val)
        }
        G.add_edge(u, v, **data)

    # add nodes' attributes to graph
    for col in gdf_nodes.columns:
        nx.set_node_attributes(G, name=col, values=gdf_nodes[col].dropna())

    return G


def add_edges_from_df(g, gdf_edges):
    reserved_columns = ["u", "v", "key"]
    attr_col_headings = [c for c in gdf_edges.columns if c not in reserved_columns]
    attribute_data = zip(*[gdf_edges[col] for col in attr_col_headings])

    for s, t, attrs in zip(gdf_edges["u"], gdf_edges["v"], attribute_data):

        g.add_edge(s, t)

        g[s][t].update(zip(attr_col_headings, attrs))


def add_nodes_from_df(g, gdf_nodes):

    gdf_nodes_dict = gdf_nodes.to_dict(orient="index")
    for n in gdf_nodes_dict:
        attrs = gdf_nodes_dict[n]
        g.add_node(n, **attrs)


def graph_from_gdfs_revisited(gdf_nodes, gdf_edges, graph_attrs=None):

    if graph_attrs is None:
        graph_attrs = {"crs": gdf_edges.crs}

    g = nx.MultiDiGraph(**graph_attrs)

    add_edges_from_df(g, gdf_edges)

    for col in ["x", "y", "geometry", "trans_mode"]:  # gdf_nodes.columns:
        nx.set_node_attributes(g, name=col, values=gdf_nodes[col].dropna())

    return g


def graph_to_gdfs2(
    G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True
):
    """
    Upgraded version of `graph_from_gdfs`, this is significantly faster.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    nodes : bool
        if True, convert graph nodes to a GeoDataFrame and return it
    edges : bool
        if True, convert graph edges to a GeoDataFrame and return it
    node_geometry : bool
        if True, create a geometry column from node x and y data
    fill_edge_geometry : bool
        if True, fill in missing edge geometry fields using nodes u and v
    Returns
    -------
    geopandas.GeoDataFrame or tuple
        gdf_nodes or gdf_edges or tuple of (gdf_nodes, gdf_edges)
    """
    crs = G.graph["crs"]

    if nodes:

        nodes, data = zip(*G.nodes(data=True))

        if node_geometry:
            # convert node x/y attributes to Points for geometry column
            geom = (Point(d["x"], d["y"]) for d in data)
            gdf_nodes = gpd.GeoDataFrame(
                data, index=nodes, crs=crs, geometry=list(geom)
            )
        else:
            gdf_nodes = gpd.GeoDataFrame(data, index=nodes)

        utils.log("Created nodes GeoDataFrame from graph")

    if edges:

        if len(G.edges) < 1:
            raise ValueError("Graph has no edges, cannot convert to a GeoDataFrame.")

        u, v, data = zip(*G.edges(data=True))

        if fill_edge_geometry:

            # subroutine to get geometry for every edge: if edge already has
            # geometry return it, otherwise create it using the incident nodes
            x_lookup = nx.get_node_attributes(G, "x")
            y_lookup = nx.get_node_attributes(G, "y")

            def make_geom(u, v, data, x=x_lookup, y=y_lookup):
                if "geometry" in data:
                    return data["geometry"]
                else:
                    return LineString((Point((x[u], y[u])), Point((x[v], y[v]))))

            geom = map(make_geom, u, v, data)
            gdf_edges = gpd.GeoDataFrame(data, crs=crs, geometry=list(geom))

        else:
            gdf_edges = gpd.GeoDataFrame(data)
            if "geometry" not in gdf_edges.columns:
                # if no edges have a geometry attribute, create null column
                gdf_edges["geometry"] = np.nan
            gdf_edges.set_geometry("geometry")
            gdf_edges.crs = crs

        # add u, v, key attributes as columns
        gdf_edges["u"] = u
        gdf_edges["v"] = v

        utils.log("Created edges GeoDataFrame from graph")

    if nodes and edges:
        return gdf_nodes, gdf_edges
    elif nodes:
        return gdf_nodes
    elif edges:
        return gdf_edges
    else:
        raise ValueError("You must request nodes or edges or both.")
