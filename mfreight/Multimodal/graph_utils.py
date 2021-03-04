import os
import re
from collections import Counter
from typing import Any, Callable, List, Tuple, TypeVar

import networkx as nx
import numpy as np
import pandas as pd
from geopy.distance import great_circle
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable
from shapely import wkt

from mfreight.utils import constants, folium_revisited
from mfreight.Multimodal import dash_plots_utils

Polygon = TypeVar("shapely.geometry.polygon.Polygon")
Graph = TypeVar("networkx.classes.multigraph.MultiGraph")
GeoDataFrame = TypeVar("geopandas.geodataframe.GeoDataFrame")
DataFrame = TypeVar("pandas.core.frame.DataFrame")
NDArray = TypeVar("numpy.ndarray")
PlotlyFig = TypeVar("plotly.graph_objs._figure.Figure")


def set_price_fun(weight):
    price_target = weight

    def price_weight_func(u, v, data):
        if data["trans_mode"] == "road":
            return data.get("dist_miles", 1)
        else:
            return data.get(price_target, 1)

    return price_weight_func


class MultimodalNet:
    def __init__(
        self,
        path_u: str = "mfreight/Multimodal/data/multimodal_G_tot_u_w_price.plk",
        payload_weight_t: float = 40.0,
    ):
        self.G_multimodal_u = nx.read_gpickle(path_u)
        self.script_dir = os.path.dirname(__file__)
        self.class1_operators = ["BNSF", "UP", "CN", "CPRS", "KCS", "CSXT", "NS"]
        self.payload_weight_t = payload_weight_t
        self.price_df = self.load_price_table()

        self.rail_edges = self.load_and_format_csv(
            self.script_dir + "/data/rail_edges.csv"
        )
        self.rail_nodes = self.load_and_format_csv(
            self.script_dir + "/data/rail_nodes.csv"
        )

        self.G_reachable_nodes = self.get_reachable_nodes()

    def load_and_format_csv(self, path: str) -> GeoDataFrame:
        df = pd.read_csv(path, index_col=0, low_memory=False)
        df["geometry"] = df["geometry"].apply(wkt.loads)
        df.columns = [
            tuple(re.findall(r"(\w+)", a)[0]) if a[0] == "(" else a for a in df.columns
        ]
        return df

    @staticmethod
    def graph_copy_nodes(G):
        copied_G = nx.Graph()
        copied_G.graph.update(G.graph)
        copied_G.add_nodes_from((n, d.copy()) for n, d in G.nodes(data=True))
        return copied_G

    def remove_rail_nodes(self, G: Graph = None):
        if G is None:
            G = self.G_reachable_nodes

        rail_nodes = [n for n, d in G.nodes(data=True) if d["trans_mode"] == "rail"]
        G.remove_nodes_from(rail_nodes)

    def get_reachable_nodes(self):
        Graph_nodes = self.graph_copy_nodes(self.G_multimodal_u)
        self.remove_rail_nodes(Graph_nodes)
        return Graph_nodes

    def get_rail_owners(self) -> List[str]:

        rail_owners_cols = [
            col
            for col in self.rail_edges.columns
            if col[:7] == "RROWNER" or col[:7] == "TRKRGHT"
        ]

        operators = pd.unique(
            self.rail_edges.loc[:, rail_owners_cols].values.ravel("K")
        )

        operators = [n for n in operators if str(n) != "nan"]

        return list(set(self.class1_operators) & set(pd.unique(operators)))

    def remove_unnecessary_nodes(self, egdes_to_keep, nodes):
        return nodes[
            ~nodes.index.isin(set(egdes_to_keep.u.values) | set(egdes_to_keep.v.values))
        ]

    def chose_operator_in_df(
        self, operators: list
    ) -> Tuple[GeoDataFrame, GeoDataFrame]:

        rail_edges = self.rail_edges.copy()
        rail_nodes = self.rail_nodes.copy()

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

        return edges_to_remove, nodes_to_remove

    def chose_operator_in_graph(
        self, operators: List[str] = ["CSXT"], G: Graph = None
    ) -> GeoDataFrame:
        if G is None:
            G = self.G_multimodal_u

        edges_to_remove, nodes_to_remove = self.chose_operator_in_df(operators)

        to_remove = zip(edges_to_remove.u, edges_to_remove.v)

        G.remove_edges_from(to_remove)

        return edges_to_remove, nodes_to_remove

    def extract_state(self, position: Tuple[float, float]):

        try:
            location = Nominatim(user_agent="green-multimodal-network").reverse(
                position, zoom=5
            )

        except TypeError:
            raise AssertionError("The chosen position is not on land")
        except GeocoderUnavailable:
            raise AssertionError(
                "Open Street Map geocoder timeout error, default pricing is shown. Please reload the page"
            )

        state = re.findall("(\w+?\s?\w+)\,", location[0])[0]
        state_code = constants.state_abbrev_map[state]

        return state_code

    def load_price_table(self):
        return pd.read_csv(
            self.script_dir + "/data/pricing.csv",
            index_col=0,
        )  # mean aggregation

    def get_price_target(
        self, departure: Tuple[float, float], arrival: Tuple[float, float]
    ):
        orig_state = self.extract_state(departure)
        dest_state = self.extract_state(arrival)

        if str((orig_state, dest_state)) in self.price_df.index:
            price_target = str((orig_state, dest_state))

        else:
            distance = great_circle(departure, arrival).miles
            if distance < 890:
                price_target = "range1"
            elif 890 <= distance < 1435:
                price_target = "range2"
            elif 1435 <= distance < 1980:
                price_target = "range3"
            elif 1980 <= distance < 2525:
                price_target = "range4"
            elif 2525 <= distance:
                price_target = "range5"

        return price_target

    @staticmethod
    def _get_weight(weight: str) -> Callable[[str], Any]:
        return lambda data: data.get(weight, 0)

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
        weights = ["trans_mode", "price", "CO2_eq_kg", "duration_h", "dist_miles"]
        route_detail = pd.DataFrame(index=range(len(path) - 1), columns=weights)
        road_rate = self.price_df.loc[price_target, "Truckload"]
        rail_rate = self.price_df.loc[price_target, "Intermodal"]

        for w in weights:
            if w == "price" and price_target:
                weight_d = self._get_weight("dist_miles")  # price_target

                route_detail[str(w)] = [
                    weight_d(G[u][v]) * road_rate
                    if G[u][v]["trans_mode"] == "road"
                    else weight_d(G[u][v]) * rail_rate
                    for u, v in zip(path[:-1], path[1:])
                ]
            else:
                weight = self._get_weight(w)
                route_detail[str(w)] = [
                    weight(G[u][v]) for u, v in zip(path[:-1], path[1:])
                ]

        route_detail["CO2_eq_kg"] = route_detail["CO2_eq_kg"] * self.payload_weight_t

        if show_entire_route:
            return route_detail

        route_summary = pd.pivot_table(
            route_detail,
            values=["price", "CO2_eq_kg", "duration_h", "dist_miles"],
            index=["trans_mode"],
            aggfunc=np.sum,
        )

        route_summary.rename(columns={"dist_miles": "distance_miles"}, inplace=True)
        route_summary = route_summary.round(2)

        if show_breakdown_by_mode:

            route_summary = route_summary.append(
                pd.DataFrame(dict(route_summary.sum()), index=["Total"])
            )
        else:
            route_summary = pd.DataFrame(dict(route_summary.sum()), index=["Total"])

        return route_summary

    def scan_route(self, route_detail: DataFrame) -> DataFrame:
        weights = ["price", "CO2_eq_kg", "duration_h", "dist_miles"]
        new_df = pd.DataFrame(columns=weights)
        new_row = pd.Series(index=weights, data=[0, 0, 0, 0])
        flag_road, flag_rail = 0, 0
        rail_counter, counter = 0, 0
        stop_list = [0]

        for idx, row in route_detail.iterrows():
            counter += 1

            if row.trans_mode == "intermodal_link":  # Ignore intermodal
                continue

            if counter == 1 and row.trans_mode == "rail":  # No road drayage
                flag_road = 1

            if row.trans_mode == "road" and flag_road == 0:
                new_row = new_row + row.loc[weights]

            elif row.trans_mode != "road" and flag_road == 0:
                flag_road = 1
                new_row["trans_mode"] = "road"
                new_df = new_df.append(new_row, ignore_index=True)
                new_row = pd.Series(index=weights, data=[0, 0, 0, 0])
                stop_list.append(idx)

            if row.trans_mode == "rail" and flag_rail == 0:
                new_row = new_row + row.loc[weights]
                rail_counter += 1

            elif row.trans_mode != "rail" and flag_rail == 0 and flag_road == 1:
                flag_rail = 1
                new_row["trans_mode"] = "rail"
                new_df = new_df.append(new_row, ignore_index=True)
                new_row = pd.Series(index=weights, data=[0, 0, 0, 0])
                stop_list.append(idx)

            if row.trans_mode == "road" and flag_road == 1:
                new_row = new_row + row.loc[weights]

            if row.trans_mode == "rail" and flag_rail == 1:
                raise AssertionError("Double Rail")

        if row.trans_mode == "road":
            new_row["trans_mode"] = "road"
        else:
            new_row["trans_mode"] = "rail"

        new_df = new_df.append(new_row, ignore_index=True).round(1)
        stop_list.append(idx)

        return new_df, rail_counter, stop_list

    def get_heuristics(self, price_target: str):
        G = self.G_multimodal_u
        min_price = min(self.price_df.loc[price_target])

        def heuristic_func_duration(u: int, v: int, G: Graph = G):
            return (
                great_circle(
                    (G.nodes[u]["y"], G.nodes[u]["x"]),
                    (G.nodes[v]["y"], G.nodes[v]["x"]),
                ).miles
                / 75
            )

        def heuristic_func_co2(u: int, v: int, G: Graph = G):
            return (
                great_circle(
                    (G.nodes[u]["y"], G.nodes[u]["x"]),
                    (G.nodes[v]["y"], G.nodes[v]["x"]),
                ).miles
                * 0.01213
            )

        def heuristic_func_price(
            u: int, v: int, G: Graph = G, min_price: float = min_price
        ):
            return (
                great_circle(
                    (G.nodes[u]["y"], G.nodes[u]["x"]),
                    (G.nodes[v]["y"], G.nodes[v]["x"]),
                ).miles
                * min_price
            )

        return {
            "duration_h": heuristic_func_duration,
            "CO2_eq_kg": heuristic_func_co2,
            "price": heuristic_func_price,
        }

    def heuristic_func_co22(self, u: int, v: int):
        G = self.G_multimodal_u
        return (
            great_circle(
                (G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])
            ).miles
            * 0.01213
        )

    def get_shortest_path(
        self,
        orig: Tuple[float, float],
        dest: Tuple[float, float],
        target_weight: str = "CO2_eq_kg",
        price_target: str = "range1",
        G: Graph = None,
    ) -> list:

        if G is None:
            G = self.G_multimodal_u

        if target_weight == "price":
            # target_weight = price_target
            target_weight = set_price_fun(price_target)

        node_orig, dist_orig = self.get_nearest_node(
            G=self.G_reachable_nodes, point=orig, return_dist=True
        )
        node_dest, dist_dest = self.get_nearest_node(
            G=self.G_reachable_nodes, point=dest, return_dist=True
        )

        (length, path) = nx.bidirectional_dijkstra(
            G=G, source=node_orig, target=node_dest, weight=target_weight
        )

        return path

    def plot_route(self, path: list, stop_list: list = None, G: Graph = None):

        if G is None:
            G = self.G_multimodal_u

        return folium_revisited.plot_route_folium(
            G,
            path,
            stop_list=stop_list,
            route_width=4,
            route_opacity=0.9,
        )

    def plot_route_visual_summary(self, scanned_route: DataFrame, path: list):

        terminal_adresses = self.get_terminal_adress(path)

        return dash_plots_utils.plot_route_detail(scanned_route, terminal_adresses)

    def chosen_path_mode(self, route_details: DataFrame) -> str:
        if "rail" in route_details.index:
            return "Multimodal"
        else:
            return "Truckload"

    def compute_all_paths(
        self,
        price_target: str,
        departure: Tuple[float, float],
        arrival: Tuple[float, float],
    ) -> DataFrame:

        summary = pd.DataFrame()
        paths = []
        targets = ["CO2_eq_kg", "duration_h"]
        for target in targets:

            path = self.get_shortest_path(departure, arrival, target_weight=target)
            paths.append(path)

            summary = summary.append(
                self.route_detail_from_graph(
                    path, show_breakdown_by_mode=False, price_target=price_target
                )
            )

        if summary.iloc[0, 0] == summary.iloc[1, 0]:  # No multimodal route
            summary = summary.iloc[0, :].set_index(pd.Index(["Truckload"]))
        else:
            summary.set_index(pd.Index(["Multimodal", "Truckload"]), inplace=True)

        return round(summary.drop(columns={"distance_miles"}), 1)

    def normalize_summary(self, summary: DataFrame) -> DataFrame:

        norm_summary = summary.copy()
        for target in summary.level_0.unique():
            norm_summary.loc[norm_summary.level_0 == target, "vals"] = (
                summary[summary.level_0 == target].vals
                / summary[summary.level_0 == target].vals.max()
            )
        return norm_summary

    def generate_bar_plot(self, summary: DataFrame, chosen_mode: str) -> PlotlyFig:

        summary = summary.copy()

        summary = summary.loc[:, ["CO2_eq_kg", "price", "duration_h"]]
        summary.rename(
            columns={
                "price": r"Price <br> [$]",
                "duration_h": r"Duration <br> [h]",
                "CO2_eq_kg": r"CO2 emissions <br> [kg_eq]",
            },
            inplace=True,
        )

        summary = (
            summary.transpose()
            .stack()
            .reset_index(inplace=False)
            .rename(columns={0: "vals"})
        )

        norm_summary = self.normalize_summary(summary)

        return dash_plots_utils.bar_plot_summary(summary, norm_summary, chosen_mode)

    def get_terminal_adress(self, path: list) -> list:
        intermodal_terminals_info = pd.read_csv(
            self.script_dir + "/data/intermodal_address.csv"
        )
        terminal_adresses = []
        for idx in range(1, len(path) - 1):
            if path[idx] < 200:
                # Road to Rail
                if (
                    self.G_multimodal_u[path[idx - 1]][path[idx]]["trans_mode"]
                    == "intermodal_link"
                ):
                    terminal_adresses.append(
                        intermodal_terminals_info.loc[path[idx], "TERM_ADDRE"]
                        + ", "
                        + intermodal_terminals_info.loc[path[idx], "TERMINAL"]
                    )
                # Rail to Road
                elif (
                    self.G_multimodal_u[path[idx]][path[idx + 1]]["trans_mode"]
                    == "intermodal_link"
                ):
                    terminal_adresses.append(
                        intermodal_terminals_info.loc[path[idx], "TERM_ADDRE"]
                        + ", "
                        + intermodal_terminals_info.loc[path[idx], "TERMINAL"]
                    )

        return terminal_adresses

    def rail_route_operators(self, path, rail_edges_counter):
        operators = []
        for u, v in zip(path[:-1], path[1:]):
            d = self.G_multimodal_u[u][v]
            for k, v in zip(d.keys(), d.values()):
                if k[:7] == "RROWNER" or k[:7] == "TRKRGHT":
                    operators.append(v)

        main_operators = []
        counter = 0
        for operator, count in Counter(operators).most_common():
            if operator in self.class1_operators:
                if counter < rail_edges_counter:
                    main_operators.append(operator)
                    counter += count
                else:
                    break

        return main_operators

    def get_nearest_node(
        self, point: Tuple[float, float], return_dist: bool = False, G: Graph = None
    ) -> int:
        if G is None:
            G = self.G_multimodal_u

        coords = ((n, d["x"], d["y"]) for n, d in G.nodes(data=True))
        df = pd.DataFrame(coords, columns=["node", "x", "y"]).set_index("node")
        df["point_lon"] = point[1]

        phi1 = np.deg2rad(df.y)
        phi2 = np.deg2rad([point[0]] * len(df))
        d_phi = phi2 - phi1

        d_theta = np.deg2rad(df.x - point[1])

        h = (
            np.sin(d_phi / 2) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(d_theta / 2) ** 2
        )
        h = np.minimum(1.0, h)  # protect against floating point errors

        arc = 2 * np.arcsin(np.sqrt(h))

        # earth_radius = 6371009 m, une m to avoid floating point errors
        dist = arc * 6371009
        node = dist.idxmin()
        if return_dist:
            return node, min(dist)
        else:
            return node
