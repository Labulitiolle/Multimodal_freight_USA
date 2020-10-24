import os
import re
from typing import TypeVar, List, Set, Dict, Tuple, Optional, Callable, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import osmnx as ox
from geopy.distance import great_circle
from geopy.geocoders import Nominatim
from mfreight.utils import astar_revisited, folium_revisited, constants, build_graph

Polygon = TypeVar("shapely.geometry.polygon.Polygon")
Graph = TypeVar("networkx.classes.multigraph.MultiGraph")
GeoDataFrame = TypeVar("geopandas.geodataframe.GeoDataFrame")
DataFrame = TypeVar("pandas.core.frame.DataFrame")
NDArray = TypeVar("numpy.ndarray")


def keep_only_intermodal(nodes: GeoDataFrame) -> GeoDataFrame:
    return nodes[nodes.trans_mode == "intermodal"]


class MultimodalNet:
    def __init__(
        self,
        path_u: str = "mfreight/Multimodal/data/multimodal_G_u.plk",
    ):
        self.G_multimodal_u = nx.read_gpickle(path_u)
        self.nodes, self.edges = ox.graph_to_gdfs(self.G_multimodal_u)
        self.script_dir = os.path.dirname(__file__)
        self.G_road = nx.read_gpickle(self.script_dir + "/data/road_G.plk")
        self.G_rail = nx.read_gpickle(self.script_dir + "/data/rail_G.plk")

    @staticmethod
    def feature_scaling(x: NDArray) -> NDArray:
        x = (x - min(x)) / (max(x) - min(x))
        return x

    def normalize_features(
        self,
        edges: GeoDataFrame,
        features: List[str] = ["price", "CO2_eq_kg", "duration_h"],
    ):

        for feature in features:
            edges[feature + "_normalized"] = self.feature_scaling(edges[feature].values)

    @staticmethod
    def set_target_feature(
        edges: GeoDataFrame,
        price_w: float = 0.33,
        duration_w: float = 0.33,
        CO2_eq_kg_w: float = 0.33,
    ):

        edges["target_feature"] = pd.eval(
            "edges.price_normalized * price_w"
            + "+ edges.CO2_eq_kg_normalized * CO2_eq_kg_w"
            + "+ edges.duration_h_normalized * duration_w"
        )


    def set_target_weight_to_edges(
        self,
        price_w: float = 0.33,
        duration_w: float = 0.33,
        CO2_eq_kg_w: float = 0.33,
    ):

        self.normalize_features(self.edges)
        self.set_target_feature(self.edges, price_w, duration_w, CO2_eq_kg_w)
        # self.G_multimodal_u = ox.graph_from_gdfs(self.nodes, self.edges)

    def get_rail_owners(self) -> List[str]:
        operators = np.array([])
        rail_owners_cols = [col for col in self.edges.columns if col[:7] == "RROWNER"]
        for col in rail_owners_cols:
            mask_is_str = [
                True if isinstance(n, str) else False for n in self.edges[col]
            ]
            operators = np.append(
                operators, pd.unique(self.edges.loc[mask_is_str, col].values.ravel("K"))
            )

        return pd.unique(operators)

    def chose_operator(self, operators: List[str] = ["CSXT"]):
        nodes, edges = self.nodes.copy(), self.edges.copy()
        rail_edges = edges[edges.trans_mode == "rail"].copy()
        rail_edges.fillna(value="None", inplace=True)
        rail_owners_cols = [col for col in rail_edges.columns if col[:7] == "RROWNER"]
        rail_rights_col = [col for col in rail_edges.columns if col[:7] == "TRKRGHT"]

        rail_operators_col = rail_owners_cols + rail_rights_col

        mask_prev = np.array([False] * len(rail_edges))
        for col in rail_operators_col:
            for operator in operators:
                mask = np.array(
                    [True if operator in n else False for n in rail_edges[col]]
                )
                mask = mask | mask_prev
                mask_prev = mask

        edges.drop(index=rail_edges[~mask_prev].index, inplace=True)
        edges.drop(
            edges.columns.difference(
                [
                    "trans_mode",
                    "length",
                    "duration_h",
                    "CO2_eq_kg",
                    "price",
                    "geometry",
                    "key",
                    "target_feature",
                    "u",
                    "v",
                ]
            ),
            axis=1,
            inplace=True,
        )
        self.G_multimodal_u = build_graph.graph_from_gdfs_revisited(nodes, edges)

        return nodes, edges

    def chose_operator_in_graph(self, nodes, edges, operators: List[str] = ["CSXT"]):
        G = build_graph.graph_from_gdfs_revisited(nodes, edges)
        to_remove = [
            (u, v)
            for u, v, d in G.edges(data=True)
            if not np.isin(operators, list(d.values())).any()
        ]
        G.remove_edges(to_remove)

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

    def set_price(self, orig: Tuple[float, float], dest: Tuple[float, float]):
        intermodal_price, truckload_price = self.get_price(orig, dest)
        self.set_price_to_edges(intermodal_price, truckload_price)

    def get_price(self, orig: Tuple[float, float], dest: Tuple[float, float]):

        orig_state = self.extract_state(orig)
        dest_state = self.extract_state(dest)

        spot_price = pd.read_csv(
            self.script_dir + "/data/spot_price.csv",
            index_col=["Origin State", "Destination State"],
        )  # mean aggregation
        contract_price = pd.read_csv(
            self.script_dir + "/data/contract_price.csv",
            index_col=["Origin State", "Destination State"],
        )

        spot_price.fillna(contract_price, inplace=True)
        spot_price = spot_price / 1.61

        try:
            intermodal_price = spot_price.loc[
                (orig_state, dest_state), "Intermodal"
            ].round(3)
            truckload_price = spot_price.loc[
                (orig_state, dest_state), "Truckload"
            ].round(3)

        except:
            intermodal_price = 0.837
            truckload_price = 1.044  # TODO DEFAULT median -> make flexible to distance

        return intermodal_price, truckload_price

    def set_price_to_graph(self, intermodal_price, truckload_price):

        for u, v in self.G_multimodal_u.edges(data=False):
            segment_length = self.G_multimodal_u[u][v][0]["length"] / 1000

            if self.G_multimodal_u[u][v][0]["trans_mode"] == "rail":
                self.G_multimodal_u[u][v][0]["price"] = (
                    intermodal_price * segment_length
                )
            elif self.G_multimodal_u[u][v][0]["trans_mode"] == "road":
                self.G_multimodal_u[u][v][0]["price"] = truckload_price * segment_length

            else:
                self.G_multimodal_u[u][v][0]["price"] = (
                    1.24 * segment_length
                )  # TODO DEFAULT

    def set_price_to_edges(self, intermodal_price, truckload_price):

        self.edges.loc[self.edges.trans_mode == "rail", "price"] = (
            intermodal_price
            * self.edges.loc[self.edges.trans_mode == "rail", "length"]
            / 1000
        )
        self.edges.loc[self.edges.trans_mode == "road", "price"] = (
            truckload_price
            * self.edges.loc[self.edges.trans_mode == "road", "length"]
            / 1000
        )
        self.edges.loc[self.edges.trans_mode == "intermodal_link", "price"] = (
            1.24
            * self.edges.loc[self.edges.trans_mode == "intermodal_link", "length"]
            / 1000
        )

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
    ) -> DataFrame:
        # The first row displays the intermodal links
        weights = ["trans_mode", "price", "CO2_eq_kg", "duration_h", "length"]
        route_detail = pd.DataFrame(index=range(len(path) - 1), columns=weights)

        for w in weights:
            weight = self._get_weight(w)
            route_detail[str(w)] = [
                weight(self.G_multimodal_u[u][v][0])
                for u, v in zip(path[:-1], path[1:])
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

    def plot_multimodal_graph(self):
        # G_road = nx.read_gpickle(self.script_dir + "/data/road_G.plk")
        # G_rail = nx.read_gpickle(self.script_dir + "/data/rail_G.plk")
        nodes_rail  = [n for n, data in self.G_multimodal_u.nodes(data=True) if data["trans_mode"]=='rail']
        G_rail = self.G_multimodal_u.subgraph(nodes_rail)
        nodes_road = [n for n, data in self.G_multimodal_u.nodes(data=True) if data["trans_mode"] == 'road']
        G_road = self.G_multimodal_u.subgraph(nodes_road)

        nodes = self.nodes.copy()
        intermodal_nodes = keep_only_intermodal(nodes)

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
            ax = ax
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

        node_orig, dist_orig = ox.get_nearest_node(
            G, orig, method="haversine", return_dist=True
        )
        node_dest, dist_dest = ox.get_nearest_node(
            G, dest, method="haversine", return_dist=True
        )

        if algo == "astar":

            weight_path = nx.astar_path_length(
                G, node_orig, node_dest, weight=target_weight
            )
            shortest_path_nodes = nx.astar_path(
                G, node_orig, node_dest, weight=target_weight
            )

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

        weight = self._get_weight("length")
        distance_highway = sum(
            [
                weight(G[u][v][0])
                for u, v in zip(shortest_path_nodes[:-1], shortest_path_nodes[1:])
            ]
        )

        dist_non_highway = dist_orig + dist_dest

        total_dist = dist_non_highway + distance_highway

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
