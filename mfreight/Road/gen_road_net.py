import os
import re
from typing import TypeVar

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

from mfreight.utils import simplify

GeoDataFrame = TypeVar("geopandas.geodataframe.GeoDataFrame")
Series = TypeVar("pandas.core.series.Series")
Polygon = TypeVar("shapely.geometry.polygon.Polygon")
Graph = TypeVar("networkx.classes.multigraph.MultiGraph")


class RoadNet:
    """
    Load datasets, compute attributes (length, CO2_eq_kg, speed_kmh, duration_h)
    and generate the road network as a graph.

    The data used is pulled from the BTS database, it only contains the highway network in the USA.

    #TODO: It will still be necessary to add the price.
    """

    def __init__(self, graph: Graph = None, kg_co2_per_tkm: float = 0.080513):
        self.trans_mode = "road"
        self.kg_co2_per_tkm = kg_co2_per_tkm
        self.G = graph
        self.script_dir = os.path.dirname(__file__)

    def load_BTS(self, bbox: Polygon = None) -> GeoDataFrame:
        """

        :param bbox: If a subset of the US rail dataset should be returned.
        e.g. bbox = Polygon([(-88, 24), (-88, 31), (-79, 31), (-79, 24)])

        :return: nodes, edges
        """
        edges = gpd.read_file(
            self.script_dir
            + "/data/National_Highway_Network-shp/National_Highway_Planning_Network.shp"
        )

        if bbox:
            edges = gpd.clip(edges, bbox)

        return edges

    @staticmethod
    def map_state_to_STFIPS():
        state_map_to_STFIPS_table = pd.read_html(
            "https://www.careerinfonet.org/links_st.asp?soccode=&stfips=&id=&nodeid=111"
        )[0].iloc[:-1, :-1]
        state_to_STFIPS_map = pd.Series(
            index=state_map_to_STFIPS_table["State Name"],
            data=state_map_to_STFIPS_table["STFIPS code"].astype("int").values,
        )
        return state_to_STFIPS_map

    @staticmethod
    def get_speed_data():
        speed_table = pd.read_html(
            "https://en.wikipedia.org/wiki/Speed_limits_in_the_United_States"
        )[1]
        speed_table["State or territory"] = speed_table[
            "State or territory"
        ].str.extract("([a-zA-Z\s]+)")
        return speed_table

    def STFIPS_to_speed_map(self):
        state_to_STFIPS_map = self.map_state_to_STFIPS()
        speed_table = self.get_speed_data()

        speed_table["STFIPS code"] = (
            speed_table["State or territory"]
            .replace(state_to_STFIPS_map)
            .astype("str")
            .str.extract("(\d+)")
        )
        speed_table["Freeway (trucks)"] = (
            speed_table["Freeway (trucks)"].str.extract("(\d+)").fillna("55")
        )

        speed_map = (
            speed_table.loc[:, ["STFIPS code", "Freeway (trucks)"]]
            .dropna(axis=0)
            .astype("int")
            .rename(columns={"Freeway (trucks)": "speed_mph"})
        )

        speed_map.set_index("STFIPS code", drop=True, inplace=True)

        speed_map_kmh = round(speed_map * 1.609344, 0).squeeze()
        return speed_map_kmh

    def add_highway_speed(self, edges):
        speed_map_kmh = self.STFIPS_to_speed_map()
        edges["speed_kmh"] = edges["STFIPS"].astype("int").replace(speed_map_kmh)

    def add_incident_nodes(self, edges):

        edges["u"] = [
            str(
                (
                    round(i.geometry.coords[:][0][0], 4),
                    round(i.geometry.coords[:][0][1], 4),
                )
            )
            for i in edges.itertuples()
        ]
        edges["v"] = [
            str(
                (
                    round(i.geometry.coords[:][-1][0], 4),
                    round(i.geometry.coords[:][-1][1], 4),
                )
            )
            for i in edges.itertuples()
        ]

    def format_gdfs(self, edges: GeoDataFrame, inplace: bool = True):
        edges.dropna(
            subset=["geometry"], inplace=True
        )  # Dropping 10 edges on the entire usa data
        self.add_incident_nodes(edges)
        self.add_highway_speed(edges)
        edges["trans_mode"] = self.trans_mode
        edges["length"] = edges["KM"] * 1000
        edges["duration_h"] = pd.eval("edges.KM / edges.speed_kmh")
        edges["CO2_eq_kg"] = pd.eval("edges.length /1000 * self.kg_co2_per_tkm")
        edges["key"] = 0

        nodes = self.gen_nodes_gdfs(edges)
        nodes["key"] = 0
        edges.drop(
            edges.columns.difference(
                [
                    "id",
                    "length",
                    "duration_h",
                    "CO2_eq_kg",
                    "u",
                    "v",
                    "STATUS",
                    "trans_mode",
                    "key",
                ]
            ),
            axis=1,
            inplace=True,
        )

        return nodes, edges

    def gen_nodes_gdfs(self, edges):
        nodes = gpd.GeoDataFrame(
            columns=["nodes_pos", "trans_mode", "x", "y", "osmid", "geometry"],
            crs="EPSG:4326",
        )

        nodes["nodes_pos"] = pd.unique(edges[["u", "v"]].values.ravel("K"))

        pattern = re.compile(r"(-?\d+.\d+)")

        coords = nodes["nodes_pos"].str.extractall(pattern).unstack(level=-1)
        coords.columns = coords.columns.droplevel()
        coords.rename(columns={0: "x", 1: "y"}, inplace=True)
        nodes["x"] = coords.x.astype(float)
        nodes["y"] = coords.y.astype(float)
        nodes["osmid"] = nodes.nodes_pos
        nodes["geometry"] = [Point(x, y) for x, y in zip(nodes.x, nodes.y)]

        nodes["trans_mode"] = self.trans_mode
        nodes["new_idx"] = range(1000000000, 1000000000 + len(nodes))
        nodes.set_index("nodes_pos", drop=True, inplace=True)
        return nodes

    def remove_attribute(self, attribute_to_remove: list = ["new_idx"]):
        for n, d in self.G.nodes(data=True):
            for att in attribute_to_remove:
                d.pop(att, None)

    def relabel_nodes(self, nodes):
        # This operation is performed once the graph has already been generated to avoid
        # using the replace function from pandas (very time consuming)

        map_ids = nodes.loc[:, "new_idx"].squeeze()
        nx.relabel_nodes(self.G, dict(map_ids), copy=False)
        self.remove_attribute()

    def keep_largest_component(self):
        largest_comp_nodes = max(nx.connected_components(self.G), key=len)
        self.G = self.G.subgraph(largest_comp_nodes).copy()

    def simplify_graph(self):
        attributes_to_sum = ["length", "CO2_eq_kg", "duration_h"]
        self.G = simplify.simplify_graph(self.G, attributes_to_sum=attributes_to_sum)
        self.G = self.G.to_undirected()
        self.keep_largest_component()

    def gen_road_graph(
        self,
        bbox: Polygon = None,
        simplified: bool = True,
        save: bool = False,
        path: str = "mfreight/multimodal/data/road_G.plk",
    ) -> Graph:

        edges = self.load_BTS(bbox)
        nodes, edges = self.format_gdfs(edges)
        self.G = ox.graph_from_gdfs(nodes, edges)
        self.relabel_nodes(nodes)

        if simplified:
            self.simplify_graph()

        if save:
            nx.write_gpickle(self.G, path)

        return self.G, nodes, edges
