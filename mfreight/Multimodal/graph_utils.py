import os
import re
from collections import Counter
from typing import Any, Callable, List, Tuple, TypeVar

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from geopy.distance import great_circle
from geopy.geocoders import Nominatim
from shapely import wkt

from mfreight.utils import constants, folium_revisited, shortest_path_revisited, plot
from mfreight.Multimodal import dash_plots_utils

Polygon = TypeVar("shapely.geometry.polygon.Polygon")
Graph = TypeVar("networkx.classes.multigraph.MultiGraph")
GeoDataFrame = TypeVar("geopandas.geodataframe.GeoDataFrame")
DataFrame = TypeVar("pandas.core.frame.DataFrame")
NDArray = TypeVar("numpy.ndarray")
PlotlyFig = TypeVar("plotly.graph_objs._figure.Figure")


class MultimodalNet:
    def __init__(
        self,
        path_u: str = "mfreight/Multimodal/data/multimodal_G_u.plk",
    ):
        self.G_multimodal_u = nx.read_gpickle(path_u)
        self.script_dir = os.path.dirname(__file__)
        self.class1_operators = ["BNSF", "UP", "CN", "CPRS", "KCS", "CSXT", "NS"]
        self.price_idx = self.get_price_idx()

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

    def chose_operator_in_graph(self, operators: List[str] = ["CSXT"]) -> GeoDataFrame:

        edges_to_remove, nodes_to_remove = self.chose_operator_in_df(operators)

        to_remove = zip(edges_to_remove.u, edges_to_remove.v)

        self.G_multimodal_u.remove_edges_from(to_remove)

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

    def get_price_idx(self):
        spot_price = self.load_price_table()
        price_idx = spot_price.index

        return price_idx

    def load_price_table(self):
        return pd.read_csv(
            self.script_dir + "/data/pricing.csv",
            index_col=0,
        )  # mean aggregation

    def get_price_target(self, departure, arrival):
        orig_state = self.extract_state(departure)
        dest_state = self.extract_state(arrival)

        if str((orig_state, dest_state)) in self.price_idx:
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

        node_orig = ox.get_nearest_node(
            self.G_reachable_nodes, orig, method="haversine"
        )
        node_dest = ox.get_nearest_node(
            self.G_reachable_nodes, dest, method="haversine"
        )

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
        weights = ["trans_mode", "price", "CO2_eq_kg", "duration_h", "dist_miles"]
        route_detail = pd.DataFrame(index=range(len(path) - 1), columns=weights)

        for w in weights:
            if w == "price" and price_target:
                # This is a work around to avoid storing floats for each edge
                # It reduces the size of the graph 211Mb -> 148Mb
                weight = self._get_weight(price_target)
                route_detail[str(w)] = [
                    weight(G[u][v])/10000 for u, v in zip(path[:-1], path[1:])
                ]
            else:
                weight = self._get_weight(w)
                route_detail[str(w)] = [
                    weight(G[u][v]) for u, v in zip(path[:-1], path[1:])
                ]

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

    def scan_route(self, route_detail):
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

    def get_heuristics(self, price_target):
        G = self.G_multimodal_u
        price_df = self.load_price_table()
        min_price = min(price_df.loc[price_target])

        def heuristic_func_duration(u, v, G=G):
            return (
                great_circle(
                    (G.nodes[u]["y"], G.nodes[u]["x"]),
                    (G.nodes[v]["y"], G.nodes[v]["x"]),
                ).miles
                / 75
            )

        def heuristic_func_co2(u, v, G=G):
            return (
                great_circle(
                    (G.nodes[u]["y"], G.nodes[u]["x"]),
                    (G.nodes[v]["y"], G.nodes[v]["x"]),
                ).miles
                * 0.01213
            )

        def heuristic_func_price(u, v, G=G, min_price=min_price):
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

    def heuristic_func_co22(self, u, v):
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
        G: Graph = None
    ):

        if G is None:
            G = self.G_multimodal_u

        if target_weight == "price":
            target_weight = price_target

        node_orig, dist_orig = ox.get_nearest_node(
            self.G_reachable_nodes, orig, method="haversine", return_dist=True
        )
        node_dest, dist_dest = ox.get_nearest_node(
            self.G_reachable_nodes, dest, method="haversine", return_dist=True
        )


        (length, path) = nx.bidirectional_dijkstra(G=G, source=node_orig, target=node_dest, weight=target_weight)

        return path

    def plot_route(self, path, stop_list=None, G=None):

        if G is None:
            G = self.G_multimodal_u

        return folium_revisited.plot_route_folium(
            G,
            path,
            stop_list=stop_list,
            route_width=4,
            route_opacity=0.9,
        )

    def plot_route_visual_summary(self, scanned_route, path):

        terminal_adresses = self.get_terminal_adress(path)

        return dash_plots_utils.plot_route_detail(scanned_route, terminal_adresses)

    def chosen_path_mode(self, route_details):
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
