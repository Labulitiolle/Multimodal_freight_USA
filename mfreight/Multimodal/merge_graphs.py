import os
from typing import TypeVar

import networkx as nx
import pandas as pd

import osmnx as ox
from geopy.distance import great_circle
from mfreight.Rail.gen_rail_net import RailNet
from mfreight.Road.gen_road_net import RoadNet
from mfreight.utils import build_graph

Polygon = TypeVar("shapely.geometry.polygon.Polygon")
Graph = TypeVar("networkx.classes.multigraph.MultiGraph")
GeoDataFrame = TypeVar("geopandas.geodataframe.GeoDataFrame")
DataFrame = TypeVar("pandas.core.frame.DataFrame")


def keep_only_intermodal(nodes: GeoDataFrame) -> GeoDataFrame:
    return nodes[nodes.trans_mode == "intermodal"]


class MergeNets:
    def __init__(self, kg_co2_per_tkm: float = 0.080513, G_multimodal_u: Graph = None):

        self.script_dir = os.path.dirname(__file__)
        self.kg_co2_per_tkm = kg_co2_per_tkm

        self.G_road, self.G_rail = None, None
        self.G_multimodal = None
        self.G_multimodal_u = G_multimodal_u

    @staticmethod
    def generate_graphs(bbox: Polygon = None) -> tuple:

        G_road = RoadNet(bbox = bbox).gen_road_graph(bbox, simplified=True, save=False)
        G_rail = RailNet(bbox = bbox).gen_rail_graph(bbox, simplified=True, save=False)

        return G_road, G_rail

    def load_preprocessed_graphs(self) -> tuple:

        G_road = nx.read_gpickle(self.script_dir + "/data/road_G.plk")
        G_rail = nx.read_gpickle(self.script_dir + "/data/rail_G.plk")

        return G_road, G_rail

    def map_intermodal_to_road(self, road_nodes: GeoDataFrame, rail_nodes: GeoDataFrame) -> DataFrame:

        intermodal_rail_nodes = keep_only_intermodal(rail_nodes)

        intermodal_to_road_map = pd.DataFrame(
            index=intermodal_rail_nodes.index, columns=["road_idx", "dist"]
        )

        road_idx = []
        dist_to_intermodal = []
        for row in intermodal_rail_nodes.itertuples():
            dist = pd.eval("(road_nodes.x-row.x)**2 + (road_nodes.y-row.y)**2")
            road_idx.append(dist.idxmin())
            dist_to_intermodal.append(
                round(
                    great_circle(
                        (row.y, row.x),
                        (
                            road_nodes.loc[dist.idxmin(), "y"],
                            road_nodes.loc[dist.idxmin(), "x"],
                        ),
                    ).km
                    * 1000,
                    2,
                )
            )

        intermodal_to_road_map["road_idx"] = road_idx
        intermodal_to_road_map["dist"] = dist_to_intermodal

        return intermodal_to_road_map

    def gen_intermodal_links(self, intermodal_to_road_map: DataFrame, road_edges: GeoDataFrame) -> DataFrame:

        intermodal_links = pd.DataFrame(
            index=range(len(intermodal_to_road_map)), columns=road_edges.columns
        )

        intermodal_links["u"] = intermodal_to_road_map.road_idx.values
        intermodal_links["v"] = intermodal_to_road_map.index.values
        intermodal_links["length"] = intermodal_to_road_map.dist.values
        intermodal_links["speed_kmh"] = 10  # DEFAULT
        intermodal_links["duration_h"] = 2  # TODO refine default value
        intermodal_links["CO2_eq_kg"] = (
            intermodal_links["length"] / 1000 * self.kg_co2_per_tkm
        )
        intermodal_links["key"] = 0
        intermodal_links["trans_mode"] = "intermodal_link"

        return intermodal_links

    def gen_intermodal_nodes(self, intermodal_to_road_map: DataFrame, road_nodes: GeoDataFrame, rail_nodes: GeoDataFrame) -> DataFrame:

        intermodal_nodes = pd.DataFrame(
            index=intermodal_to_road_map.index, columns=road_nodes.columns
        )

        x, y, geometry, zip_code = [], [], [], []

        for row in intermodal_to_road_map.itertuples():
            x.append(rail_nodes.loc[row.Index, "x"])
            y.append(rail_nodes.loc[row.Index, "y"])
            geometry.append(rail_nodes.loc[row.Index, "geometry"])
            zip_code.append(rail_nodes.loc[row.Index, "STCYFIPS"])

        intermodal_nodes["x"] = x
        intermodal_nodes["y"] = y
        intermodal_nodes["geometry"] = geometry
        intermodal_nodes["STCYFIPS"] = zip_code
        intermodal_nodes["trans_mode"] = "intermodal"
        intermodal_nodes["key"] = 0

        return intermodal_nodes

    def link_road_to_rail(self, road_nodes: GeoDataFrame, rail_nodes: GeoDataFrame, road_edges: GeoDataFrame) -> tuple:

        intermodal_to_road_map = self.map_intermodal_to_road(road_nodes, rail_nodes)

        intermodal_links = self.gen_intermodal_links(intermodal_to_road_map, road_edges)
        road_edges = road_edges.append(intermodal_links, ignore_index=True)

        intermodal_nodes = self.gen_intermodal_nodes(
            intermodal_to_road_map, road_nodes, rail_nodes
        )
        road_nodes = road_nodes.append(intermodal_nodes)

        return road_nodes, road_edges

    def merge_networks(
        self,
        import_preprocesses_graphs: bool = True,
        save: bool = True,
        path: str = "mfreight/multimodal/data/multimodal_G.plk",  # TODO relative path
        path_u: str = "mfreight/multimodal/data/multimodal_G_u.plk",
    ):

        if import_preprocesses_graphs:
            self.G_road, self.G_rail = self.load_preprocessed_graphs()

        else:
            self.G_road, self.G_rail = self.generate_graphs()

        road_nodes, road_edges = ox.graph_to_gdfs(self.G_road)
        rail_nodes, rail_edges = ox.graph_to_gdfs(self.G_rail)

        road_nodes, road_edges = self.link_road_to_rail(
            road_nodes, rail_nodes, road_edges
        )

        G_road_w_link = ox.graph_from_gdfs(road_nodes, road_edges)

        self.G_multimodal = nx.compose(G_road_w_link, self.G_rail)
        nodes, edges = ox.graph_to_gdfs(self.G_multimodal)
        # edges = edges[edges.key != 1] TODO There are 12 multi edges, not necessary to be removed
        ox.graph_from_gdfs(nodes, edges)
        self.G_multimodal_u = build_graph.graph_from_gdfs(
            nodes, edges, undirected=True
        )  # TODO redesigned osmnx function to make the graph bidirectionnal

        if save:
            nx.write_gpickle(self.G_multimodal.to_undirected(), path)
            nx.write_gpickle(self.G_multimodal_u, path_u)



