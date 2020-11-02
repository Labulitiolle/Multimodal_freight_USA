import os
import re
import time
from typing import Any, Callable, List, Tuple, TypeVar

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from geopy.distance import great_circle
from geopy.geocoders import Nominatim
from shapely import wkt
from shapely.geometry import LineString, Point

from mfreight.utils import astar_revisited, build_graph, constants, folium_revisited

Polygon = TypeVar("shapely.geometry.polygon.Polygon")
Graph = TypeVar("networkx.classes.multigraph.MultiGraph")
GeoDataFrame = TypeVar("geopandas.geodataframe.GeoDataFrame")
DataFrame = TypeVar("pandas.core.frame.DataFrame")
NDArray = TypeVar("numpy.ndarray")


class MultimodalNet:
    def __init__(
        self,
        path_u: str = "mfreight/Multimodal/data/multimodal_G_u.plk",
    ):
        self.G_multimodal_u = nx.read_gpickle(path_u)
        print('length:', len(self.G_multimodal_u))
        self.script_dir = os.path.dirname(__file__)
        self.class1_operators = ["BNSF", "UP", "CN", "CPRS", "KCS", "CSXT", "NS"]

        self.rail_edges = self.load_and_format_csv(
            self.script_dir + "/data/rail_edges.csv"
        )
        self.rail_nodes = self.load_and_format_csv(
            self.script_dir + "/data/rail_nodes.csv"
        )

    def load_and_format_csv(self, path: str) -> GeoDataFrame:
        df = pd.read_csv(path, index_col=0)
        df["geometry"] = df["geometry"].apply(wkt.loads)
        df.columns = [
            tuple(re.findall(r"(\w+)", a)[0]) if a[0] == "(" else a for a in df.columns
        ]
        return df

    def get_rail_owners(self) -> List[str]:
        operators = np.array([])
        rail_owners_cols = [
            col
            for col in self.rail_edges.columns
            if col[:7] == "RROWNER" or col[:7] == "TRKRGHT"
        ]
        for col in rail_owners_cols:
            mask_is_str = [
                True if isinstance(n, str) else False for n in self.rail_edges[col]
            ]
            operators = pd.unique(
                self.rail_edges.loc[mask_is_str, rail_owners_cols].values.ravel("K")
            )

        return list(set(self.class1_operators) & set(pd.unique(operators)))

    def remove_unnecessary_nodes(self, egdes_to_keep, nodes):
        return nodes[
            ~nodes.index.isin(set(egdes_to_keep.u.values) | set(egdes_to_keep.v.values))
        ]

    def chose_operator_in_df(
        self, operators: list
    ) -> Tuple[GeoDataFrame, GeoDataFrame]:
        start = time.time()
        rail_edges = self.rail_edges.copy()
        rail_nodes = self.rail_nodes.copy()
        print(f"22Elapsed time: {time.time()-start}")
        start = time.time()
        rail_edges.fillna(value="None", inplace=True)
        rail_owners_cols = [col for col in rail_edges.columns if col[:7] == "RROWNER"]
        rail_rights_col = [col for col in rail_edges.columns if col[:7] == "TRKRGHT"]

        rail_operators_col = rail_owners_cols + rail_rights_col

        mask_prev = np.array([False] * len(rail_edges))

        for operator in operators:
            mask = np.array(
                [
                    True if operator in n else False
                    for n in rail_edges[rail_operators_col].itertuples()
                ]
            )
            mask = mask | mask_prev
            mask_prev = mask

        egdes_to_keep = rail_edges.loc[mask_prev, :]
        edges_to_remove = rail_edges.drop(index=egdes_to_keep.index)

        nodes_to_remove = self.remove_unnecessary_nodes(egdes_to_keep, rail_nodes)

        print(f"33Elapsed time: {time.time() - start}")
        return edges_to_remove, nodes_to_remove

    def chose_operator_in_graph(self, operators: List[str] = ["CSXT"]) -> GeoDataFrame:

        start = time.time()
        edges_to_remove, nodes_to_remove = self.chose_operator_in_df(operators)

        to_remove = zip(edges_to_remove.u, edges_to_remove.v)

        self.G_multimodal_u.remove_edges_from(to_remove)

        print(f"44Elapsed time: {time.time() - start}")
        return edges_to_remove, nodes_to_remove

    def extract_state(self, position: Tuple[float, float]):
        try:
            location = Nominatim(user_agent="green-multimodal-network").reverse(
                position, zoom=5
            )
        except:
            raise AssertionError(
                "The chosen position is not on land"
            )  # TODO improve the dash error message
        state = re.findall("(\w+)\,", location[0])[0]
        state_code = constants.state_abbrev_map[state]

        return state_code

    def set_price_to_graph(self):

        spot_price = self.build_price_table()

        price_idx = spot_price.index.to_flat_index()

        for u, v, d in self.G_multimodal_u.edges(data=True):
            prices_for_segment = d["length"] * spot_price

            if d["trans_mode"] == "road":
                self.G_multimodal_u[u][v][0].update(
                    zip(price_idx, prices_for_segment["Truckload"])
                )

            elif d["trans_mode"] == "rail":
                self.G_multimodal_u[u][v][0].update(
                    zip(price_idx, prices_for_segment["Intermodal"])
                )
            else:
                self.G_multimodal_u[u][v][0].update(
                    zip(price_idx, prices_for_segment["Intermodal"])
                )

    def build_price_table(self):
        spot_price = pd.read_csv(
            self.script_dir + "/data/spot_price.csv",
            index_col=["Origin State", "Destination State"],
        )  # mean aggregation
        contract_price = pd.read_csv(
            self.script_dir + "/data/contract_price.csv",
            index_col=["Origin State", "Destination State"],
        )

        spot_price.fillna(contract_price, inplace=True)
        spot_price.fillna(method="ffill")  # TODO improve padding
        spot_price = spot_price / 1.61

        default_price = pd.DataFrame(
            index=[("default", "default")],
            columns=["Intermodal", "Truckload"],
            data=[[0.837, 1.044]],
        )

        spot_price = spot_price.append(default_price)

        return spot_price

    @staticmethod
    def _get_weight(weight: str) -> Callable[[str], Any]:
        return lambda data: data.get(weight, 0)

    def route_detail_from_orig_dest(
        self,
        orig: Tuple[float, float],
        dest: Tuple[float, float],
        target_weight: str = "CO2_eq_kg",
        G: Graph = None,
        show_entire_route: bool = False,
    ) -> DataFrame:
        if G is None:
            G = self.G_multimodal_u

        node_orig = ox.get_nearest_node(G, orig, method="haversine")
        node_dest = ox.get_nearest_node(G, dest, method="haversine")

        shortest_path_nx = nx.astar_path(G, node_orig, node_dest, weight=target_weight)

        return self.route_detail_from_graph(
            shortest_path_nx, show_entire_route=show_entire_route
        )

    def route_detail_from_graph(
        self,
        path: List[int],
        show_breakdown_by_mode: bool = True,
        show_entire_route: bool = False,
        G: Graph = None,
        price_target=None,
    ) -> DataFrame:
        if G is None:
            G = self.G_multimodal_u
        # The first row displays the intermodal links
        weights = ["trans_mode", "price", "CO2_eq_kg", "duration_h", "length"]
        route_detail = pd.DataFrame(index=range(len(path) - 1), columns=weights)

        for w in weights:
            if w == "price" and price_target:
                weight = self._get_weight(price_target)
                route_detail[str(w)] = [
                    weight(G[u][v][0]) for u, v in zip(path[:-1], path[1:])
                ]
            else:
                weight = self._get_weight(w)
                route_detail[str(w)] = [
                    weight(G[u][v][0]) for u, v in zip(path[:-1], path[1:])
                ]

        if show_entire_route:
            return route_detail

        else:
            route_summary = pd.pivot_table(
                route_detail,
                values=["price", "CO2_eq_kg", "duration_h", "length"],
                index=["trans_mode"],
                aggfunc=np.sum,
            )
            route_summary["length"] = route_summary["length"] / 1000
            route_summary.rename(columns={"length": "distance_km"}, inplace=True)
            route_summary = route_summary.round(2)

            if show_breakdown_by_mode:

                route_summary = route_summary.append(
                    pd.DataFrame(dict(route_summary.sum()), index=["Total"])
                )
            else:
                route_summary = pd.DataFrame(dict(route_summary.sum()), index=["Total"])

            return route_summary  # , list(route_detail.trans_mode).count("intermodal_link") TODO do I need this ?

    def plot_multimodal_graph(self, G=None, bbox=None):
        """

        :param bbox: If a subset of teh graph should be returned.
        e.g. bbox = (west,north,east,south)

        :return: nodes, edges
        """
        if G is None:
            G = self.G_multimodal_u

        nodes_road, nodes_rail = [], []
        nodes_intermodal_geometry = []
        if bbox:
            for n, data in G.nodes(data=True):
                if bbox[0] < data["x"] < bbox[2] and bbox[3] < data["y"] < bbox[1]:
                    if data["trans_mode"] == "road":
                        nodes_road.append(n)
                    elif (
                        data["trans_mode"] == "rail"
                        or data["trans_mode"] == "intermodal"
                    ):
                        nodes_rail.append(n)
                    if data["trans_mode"] == "intermodal":
                        nodes_intermodal_geometry.append(data["geometry"])
        else:
            for n, data in G.nodes(data=True):
                if data["trans_mode"] == "road":
                    nodes_road.append(n)
                elif data["trans_mode"] == "rail" or data["trans_mode"] == "intermodal":
                    nodes_rail.append(n)
                if data["trans_mode"] == "intermodal":
                    nodes_intermodal_geometry.append(data["geometry"])

        G_road = G.subgraph(nodes_road)
        G_rail = G.subgraph(nodes_rail)
        intermodal_nodes = gpd.GeoDataFrame(
            {"geometry": nodes_intermodal_geometry}, crs="epsg:4326"
        )

        fig, ax = plt.subplots(figsize=(20, 10))
        ox.plot_graph(
            G_rail,
            edge_color="green",
            edge_linewidth=1,
            edge_alpha=0.5,
            node_color="yellow",
            node_size=1,
            node_alpha=0.1,
            bgcolor="white",
            show=False,
            ax=ax,
        )
        intermodal_nodes.plot(ax=ax, color="red")
        ox.plot_graph(
            G_road,
            edge_color="blue",
            edge_linewidth=1,
            edge_alpha=0.1,
            node_color="blue",
            bgcolor="white",
            node_size=0.01,
            node_alpha=0.2,
            show=True,
            ax=ax,
        )

    def plot_route(
        self,
        orig: Tuple[float, float],
        dest: Tuple[float, float],
        target_weight: str = "CO2_eq_kg",
        G: Graph = None,
        orig_dest_size: int = 100,
        algo: str = "astar",
        folium: bool = False,
    ):
        if G is None:
            G = self.G_multimodal_u

        start = time.time()
        node_orig, dist_orig = ox.get_nearest_node(
            G, orig, method="haversine", return_dist=True
        )
        node_dest, dist_dest = ox.get_nearest_node(
            G, dest, method="haversine", return_dist=True
        )

        print(f"nearest_node Elapsed time: {time.time() - start}")
        if algo == "astar":
            start = time.time()

            weight_path = nx.astar_path_length(
                G, node_orig, node_dest, weight=target_weight
            )
            shortest_path_nodes = nx.astar_path(
                G, node_orig, node_dest, weight=target_weight
            )
            print(f"A* Elapsed time: {time.time() - start}")

        elif algo == "dijkstra":

            weight_path = nx.dijkstra_path_length(
                G, node_orig, node_dest, weight=target_weight
            )
            shortest_path_nodes = nx.dijkstra_path(
                G, node_orig, node_dest, weight=target_weight
            )

        else:
            raise AssertionError(
                f'The parameter "algo" can either be "astar" or "dijkstra", not {algo}'
            )

        if folium:
            return (
                folium_revisited.plot_route_folium(
                    G,
                    shortest_path_nodes,
                    route_width=4,
                    route_opacity=0.9,
                ),
                shortest_path_nodes,
            )

        else:
            weight = self._get_weight("length")
            distance_highway = sum(
                [
                    weight(G[u][v][0])
                    for u, v in zip(shortest_path_nodes[:-1], shortest_path_nodes[1:])
                ]
            )

            dist_non_highway = dist_orig + dist_dest

            total_dist = dist_non_highway + distance_highway

            fig, ax = plt.subplots(figsize=(15, 7))
            ax.scatter(orig[1], orig[0], marker="x", s=orig_dest_size, zorder=5)
            ax.scatter(dest[1], dest[0], marker="x", s=orig_dest_size, zorder=10)
            ox.plot_graph(
                G,
                edge_color="#bfbfbf",
                node_color="#595959",
                bgcolor="w",
                ax=ax,
                show=False,
            )
            ox.plot_graph_route(
                G,
                shortest_path_nodes,
                route_color="g",
                route_linewidth=4,
                route_alpha=0.9,
                orig_dest_size=100,
                ax=ax,
                show=True,
            )

            self.print_graph_info(total_dist, weight_path, target_weight)

    @staticmethod
    def print_graph_info(
        total_dist: Tuple[float, float],
        weight_path: Tuple[float, float],
        target_weight: str = "CO2_eq_kg",
    ):

        if target_weight == "length":

            print(f"Distance {total_dist / 1000} [km]")

        elif target_weight == "duration_h":
            if weight_path > 1:
                print(f"Duration {weight_path} [h] \nDistance {total_dist / 1000} [km]")

            else:
                print(
                    f"Duration {weight_path * 60} [min] \nDistance {total_dist / 1000} [km]"
                )

        elif target_weight == "CO2_eq_kg":
            print(
                f"{str(target_weight)}: {weight_path} \nDistance {total_dist / 1000} [km]"
            )

        else:
            print(f"{str(target_weight)}: {target_weight}")

    def heuristic_func(self, u: int, v: int) -> float:
        return (
            great_circle(
                (self.G_multimodal_u.nodes[u]["y"], self.G_multimodal_u.nodes[u]["x"]),
                (self.G_multimodal_u.nodes[v]["y"], self.G_multimodal_u.nodes[v]["x"]),
            ).km
            * 1000
        )

    def plot_astar_exploration(
        self,
        orig: Tuple[float, float],
        dest: Tuple[float, float],
        target_weight: str = "CO2_eq_kg",
        heuristic: Callable[[int, int], float] = None,
        orig_dest_size: int = 100,
    ):
        if not heuristic:
            heuristic = self.heuristic_func()

        node_orig, dist_orig = ox.get_nearest_node(
            self.G_multimodal_u, orig, method="haversine", return_dist=True
        )
        node_dest, dist_dest = ox.get_nearest_node(
            self.G_multimodal_u, dest, method="haversine", return_dist=True
        )

        dist_path = nx.astar_path_length(
            self.G_multimodal_u, node_orig, node_dest, weight="length"
        )
        weight_path = nx.astar_path_length(
            self.G_multimodal_u, node_orig, node_dest, weight=target_weight
        )

        shortest_path_nodes, explored = astar_revisited.astar_path(
            self.G_multimodal_u,
            node_orig,
            node_dest,
            weight=target_weight,
            heuristic=heuristic,
        )

        distance_highway = dist_path

        dist_non_highway = dist_orig + dist_dest

        total_dist = dist_non_highway + distance_highway

        explored_nodes = set(explored.keys())
        explored_nodes.update(explored.values())
        explored_nodes.remove(None)
        nodes = self.nodes.copy()
        explored_nodes = nodes.loc[list(explored_nodes), :]

        fig, ax = plt.subplots(figsize=(15, 7))

        ax.scatter(orig[1], orig[0], marker="x", s=orig_dest_size, zorder=5)
        ax.scatter(dest[1], dest[0], marker="x", s=orig_dest_size, zorder=10)
        ax.scatter(
            explored_nodes.x, explored_nodes.y, marker="o", color="red", s=2, zorder=100
        )

        ox.plot_graph(
            self.G_multimodal_u,
            edge_color="#bfbfbf",
            node_color="#595959",
            bgcolor="w",
            ax=ax,
            show=False,
        )
        ox.plot_graph_route(
            self.G_multimodal_u,
            shortest_path_nodes,
            route_color="g",
            route_linewidth=4,
            route_alpha=0.9,
            orig_dest_size=100,
            ax=ax,
        )

        self.print_graph_info(total_dist, weight_path, target_weight)
