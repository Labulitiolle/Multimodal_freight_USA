import os
import re
from typing import TypeVar

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd

from mfreight.utils import simplify

GeoDataFrame = TypeVar("geopandas.geodataframe.GeoDataFrame")
Series = TypeVar("pandas.core.series.Series")
Polygon = TypeVar("shapely.geometry.polygon.Polygon")
Graph = TypeVar("networkx.classes.multigraph.MultiGraph")


class RailNet:
    """
    Load dataset, add the intermodal facilities, compute attributes (length, CO2_eq_kg, duration_h)
    and generate the rail network as a graph.

    #TODO: It will still be necessary to add the price
    """

    def __init__(self, graph: Graph = None, kg_co2_per_tkm=0.0195254):
        self.trans_mode = "rail"
        self.kg_co2_per_tkm = kg_co2_per_tkm
        self.track_to_speed_map = {
            0: 10,
            1: 16,
            2: 40,
            3: 64,
            4: 97,
            5: 130,
            6: 180,
            13: 97,
            None: 10
        }
        self.G = graph
        self.script_dir = os.path.dirname(__file__)

    def load_BTS(self, bbox: Polygon = None) -> tuple:

        edges = gpd.read_file(
            self.script_dir
            + "/rail_data/North_American_Rail_Lines-shp/North_American_Rail_Lines.shp"
        )
        nodes = gpd.read_file(
            self.script_dir
            + "/rail_data/North_American_Rail_Nodes-shp/North_American_Rail_Nodes.shp"
        )

        if bbox:
            edges = gpd.clip(edges, bbox)

        return nodes, edges

    @staticmethod
    def keep_only_valid_usa_rail(nodes: GeoDataFrame, edges: GeoDataFrame):

        edges.drop(edges[~edges.eval("(NET == 'I' | NET == 'M') &  COUNTRY == 'US'")].index, inplace=True)
        nodes.drop(nodes[~nodes.FRANODEID.isin(list(edges.u.values) + list(edges.v.values))].index, inplace=True)
        nodes.set_index("FRANODEID", drop=False, inplace=True)

    def format_gpdfs(self, nodes: GeoDataFrame, edges: GeoDataFrame):

        edges.rename(columns={"FRFRANODE": "u", "TOFRANODE": "v"}, inplace=True)

        edges["trans_mode"] = self.trans_mode
        edges["length"] = edges["KM"] * 1000
        edges["CO2_eq_kg"] = pd.eval("edges.length /1000 * self.kg_co2_per_tkm")
        edges["key"] = 0

        nodes["trans_mode"] = self.trans_mode
        nodes["key"] = 0

        self.keep_only_valid_usa_rail(nodes, edges)
        self.add_speed_duration(edges)

        edges.drop(
            columns=[
                "YARDNAME",
                "CNTYFIPS",
                "STCNTYFIPS",
                "STATEAB",
                "COUNTRY",
                "FRAREGION",
                "SUBDIV",
                "CARDDIRECT",
                "MILES",
                "KM",
                "ShapeSTLen",
                "geometry",
            ],
            inplace=True,
        )
        nodes.drop(
            columns=[
                "OBJECTID",
                "COUNTRY",
                "STFIPS",
                "CTYFIPS",
                "FRAREGION",
                "BNDRY",
                "PASSNGR",
                "PASSNGRSTN",
            ],
            inplace=True,
        )

        nodes.set_index("FRANODEID", drop=False, inplace=True)
        self.add_x_y_pos(nodes)

    def add_speed_duration(self, edges: GeoDataFrame):
        edges["speed_kmh"] = edges.TRACKS.replace(self.track_to_speed_map)
        edges["duration_h"] = pd.eval("edges.length * 1000 / edges.speed_kmh")

    def add_x_y_pos(self, gdf: GeoDataFrame):

        pattern = re.compile(r"(-?\d+\.?\d+)")

        coords = gdf.geometry.astype("str").str.extractall(pattern).unstack(level=-1)
        coords.columns = coords.columns.droplevel()
        coords.rename(columns={0: "x", 1: "y"}, inplace=True)
        gdf["x"] = coords.x.astype(float)
        gdf["y"] = coords.y.astype(float)

    def load_intermodal_facilities(self) -> GeoDataFrame:
        return gpd.read_file(
            self.script_dir
            + "/rail_data/Intermodal stations/Intermodal_Freight_Facilities_RailTOFCCOFC.shp"
        )

    @staticmethod
    def map_rail_to_intermodal_nodes(
        intermodal_nodes: GeoDataFrame, nodes: GeoDataFrame
    ) -> Series:
        rail_node = []
        for row in intermodal_nodes.itertuples():
            dist = pd.eval("(nodes.x-row.x)**2 + (nodes.y-row.y)**2")
            rail_node.append(dist.idxmin())

        rail_to_intermodal_map = pd.Series(
            intermodal_nodes.OBJECTID.values, index=rail_node
        )

        return rail_to_intermodal_map

    def add_intermodal_nodes(self, nodes: GeoDataFrame, edges: GeoDataFrame):
        intermodal_nodes = self.load_intermodal_facilities()

        self.add_x_y_pos(intermodal_nodes)

        rail_to_intermodal_map = self.map_rail_to_intermodal_nodes(
            intermodal_nodes, nodes
        )
        nodes.loc[:, "FRANODEID"].replace(rail_to_intermodal_map, inplace=True)
        nodes.loc[nodes.FRANODEID < 200, "trans_mode"] = "intermodal"

        edges.loc[:, ["u", "v"]] = edges.loc[:, ["u", "v"]].replace(
            rail_to_intermodal_map
        )

    def keep_largest_component(self):
        largest_comp_nodes = max(nx.connected_components(self.G), key=len)
        self.G = self.G.subgraph(largest_comp_nodes).copy()

    def simplify_graph(self, nodes: GeoDataFrame):
        nodes_to_keep = list(nodes[nodes.index < 200].index)
        attributes_to_sum = ["length", "CO2_eq_kg", "duration_h"]
        self.G = simplify.simplify_graph(
            self.G, attributes_to_sum=attributes_to_sum, nodes_to_keep=nodes_to_keep
        )

        self.G = self.G.to_undirected()
        self.keep_largest_component()

    def gen_rail_graph(
        self,
        bbox: Polygon = None,
        simplified: bool = True,
        save: bool = False,
        path: str = "mfreight/multimodal/data/rail_G.plk",
    ) -> Graph:

        nodes, edges = self.load_BTS(bbox)
        self.format_gpdfs(nodes, edges)
        self.add_intermodal_nodes(nodes, edges)
        self.G = ox.graph_from_gdfs(nodes, edges)

        if simplified:
            self.simplify_graph(nodes)

        if save:
            nx.write_gpickle(self.G, path)

        return self.G, nodes, edges