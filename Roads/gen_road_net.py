import os
import re
from typing import TypeVar

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

from utils import simplify

GeoDataFrame = TypeVar("geopandas.geodataframe.GeoDataFrame")
Series = TypeVar("pandas.core.series.Series")
Polygon = TypeVar("shapely.geometry.polygon.Polygon")
Graph = TypeVar("networkx.classes.multigraph.MultiGraph")


class RoadNet:
    def __init__(self, kg_co2_per_tkm=0.080513):
        self.mode = "road"
        self.kg_co2_per_tkm = kg_co2_per_tkm
        self.G = None
        self.script_dir = os.path.dirname(__file__)

    def load_BTS(self, bbox: Polygon = None) -> tuple:
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
        )[0].iloc[:-1, :]
        state_map_to_STFIPS_map = pd.Series(
            index=state_map_to_STFIPS_table["State Name"],
            data=state_map_to_STFIPS_table["STFIPS code"].astype("int").values,
        )
        return state_map_to_STFIPS_map

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
        state_map_to_STFIPS_map = self.map_state_to_STFIPS()
        speed_table = self.get_speed_data()

        speed_table["STFIPS code"] = (
            speed_table["State or territory"]
            .replace(state_map_to_STFIPS_map)
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

        return edges

    def format_gdfs(self, edges: GeoDataFrame, inplace: bool = True) -> tuple:

        if inplace:
            edges_new = edges
        else:
            edges_new = edges.copy()

        self.add_incident_nodes(edges_new)
        self.add_highway_speed(edges_new)
        edges_new["tag"] = self.mode
        edges_new["length"] = edges_new["KM"] * 1000
        edges_new["duration_h"] = pd.eval("edges_new.KM / edges_new.speed_kmh")
        edges_new["CO2_eq_kg"] = pd.eval("edges_new.length /1000 * self.kg_co2_per_tkm")
        edges_new["key"] = 0

        nodes_new = self.gen_nodes_gdfs(edges)
        nodes_new["key"] = 0
        edges_new.drop(
            edges_new.columns.difference(
                [
                    "id",
                    "length",
                    "duration_h",
                    "CO2_eq_kg",
                    "u",
                    "v",
                    "STATUS",
                    "tag",
                    "key",
                ]
            ),
            axis=1,
            inplace=True,
        )

        return nodes_new, edges_new

    def gen_nodes_gdfs(self, edges):
        nodes = gpd.GeoDataFrame(
            columns=["nodes_pos", "tag", "x", "y", "osmid", "geometry"], crs="EPSG:4326"
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

        nodes["tag"] = self.mode
        nodes["new_idx"] = range(1000000000, 1000000000 + len(nodes))
        nodes.set_index("nodes_pos", drop=True, inplace=True)

        return nodes

    def relabel_nodes(self, nodes):
        map_ids = nodes.loc[:, "new_idx"].squeeze()

        nx.relabel_nodes(self.G, dict(map_ids), copy=False)

    def keep_largest_component(self):
        largest_comp_nodes = max(nx.connected_components(self.G), key=len)
        self.G = self.G.subgraph(largest_comp_nodes).copy()

    def simplify_graph(self, nodes: GeoDataFrame):
        attributes_to_sum = ["length", "CO2_eq_kg", "duration_h"]
        self.G = simplify.simplify_graph(self.G, attributes_to_sum=attributes_to_sum)

    def gen_road_graph(self, bbox: Polygon = None, simplified: bool = True) -> Graph:

        edges = self.load_BTS(bbox)
        nodes, edges = self.format_gdfs(edges)
        self.G = ox.graph_from_gdfs(nodes, edges)
        self.relabel_nodes(nodes)

        if simplified:
            self.simplify_graph(nodes)
            self.G = self.G.to_undirected()
            self.keep_largest_component()

        return self.G, nodes, edges
