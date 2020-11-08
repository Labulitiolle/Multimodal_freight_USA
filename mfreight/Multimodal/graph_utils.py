import os
import re
import time
from collections import Counter
from typing import Any, Callable, List, Tuple, TypeVar

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import plotly.graph_objects as go
from geopy.distance import great_circle
from geopy.geocoders import Nominatim
from shapely import wkt

from mfreight.utils import astar_revisited, constants, folium_revisited

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

        self.rail_edges = self.load_and_format_csv(
            self.script_dir + "/data/rail_edges.csv"
        )
        self.rail_nodes = self.load_and_format_csv(
            self.script_dir + "/data/rail_nodes.csv"
        )

    def load_and_format_csv(self, path: str) -> GeoDataFrame:
        df = pd.read_csv(path, index_col=0, low_memory=False)
        df["geometry"] = df["geometry"].apply(wkt.loads)
        df.columns = [
            tuple(re.findall(r"(\w+)", a)[0]) if a[0] == "(" else a for a in df.columns
        ]
        return df

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

    def set_price_to_graph(self):

        spot_price = self.load_price_table()

        price_idx = spot_price.index

        for u, v, d in self.G_multimodal_u.edges(data=True):
            prices_for_segment = d["dist_miles"] * spot_price

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

        return price_idx

    def load_price_table(self):
        return pd.read_csv(
            self.script_dir + "/data/pricing.csv",
            index_col=0,
        )  # mean aggregation

    def get_price_target(self, departure, arrival, price_idx):
        orig_state = self.extract_state(departure)
        dest_state = self.extract_state(arrival)

        if (orig_state, dest_state) in price_idx:
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
        weights = ["trans_mode", "price", "CO2_eq_kg", "duration_h", "dist_miles"]
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

    def plot_route_detail(self, scanned_route, path):

        scanned_route_norm = scanned_route.loc[
            :, ["price", "CO2_eq_kg", "duration_h", "dist_miles"]
        ] / scanned_route.loc[
            :, ["price", "CO2_eq_kg", "duration_h", "dist_miles"]
        ].sum(
            axis=0
        )
        if len(scanned_route) == 3:
            terminal_adresses = self.get_terminal_adress(path)
            first_leg = 400 * scanned_route_norm.loc[0, "dist_miles"]
            second_leg = 400 * scanned_route_norm.loc[1, "dist_miles"]
            third_leg = 400 * scanned_route_norm.loc[2, "dist_miles"]

            if third_leg < 12:
                third_leg = 12
            if first_leg < 12:
                first_leg = 12

            marker_space = 10
            fig = go.Figure(
                go.Scatter(
                    x=[0],
                    y=[0.5],
                    marker=dict(size=20, color="black"),
                    text="Departure",
                    hoverinfo="text",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[
                        marker_space,
                        marker_space,
                        first_leg + marker_space - 10,
                        first_leg + marker_space,
                        first_leg + marker_space - 10,
                    ],
                    y=[0, 1, 1, 0.5, 0],
                    fill="toself",
                    fillcolor="grey",
                    line_width=0.1,
                    marker=dict(size=0.1),
                    hoveron="fills",
                    text=f"<b>Mode:</b> {scanned_route.loc[0, 'trans_mode']} <br>Price: {scanned_route.loc[0, 'price']}, CO2: {scanned_route.loc[0, 'CO2_eq_kg']}, dist: {scanned_route.loc[0, 'dist_miles']}",
                    hoverinfo="text",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[first_leg + marker_space * 2],
                    y=[0.5],
                    marker=dict(size=20, color="black"),
                    text=f"<b>Terminal:</b> <br>{terminal_adresses[1]}",
                    hoverinfo='text'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[
                        first_leg + marker_space * 3,
                        first_leg + marker_space * 3,
                        first_leg + marker_space * 3 + second_leg - 10,
                        first_leg + marker_space * 3 + second_leg,
                        first_leg + marker_space * 3 + second_leg - 10,
                    ],
                    y=[0, 1, 1, 0.5, 0],
                    fill="toself",
                    fillcolor="green",
                    line_width=0.1,
                    marker=dict(size=0.1),
                    hoveron="fills",
                    text=f"<b>Mode:</b> {scanned_route.loc[1, 'trans_mode']} <br>Price: {scanned_route.loc[1, 'price']}, CO2: {scanned_route.loc[1, 'CO2_eq_kg']}, dist: {scanned_route.loc[1, 'dist_miles']}",
                    hoverinfo="text",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[first_leg + marker_space * 4 + second_leg],
                    y=[0.5],
                    marker=dict(size=20, color="black"),
                    text=f"<b>Terminal:</b> <br>{terminal_adresses[1]}",
                    hoverinfo='text'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[
                        first_leg + marker_space * 5 + second_leg,
                        first_leg + marker_space * 5 + second_leg,
                        first_leg + marker_space * 5 + second_leg + third_leg - 10,
                        first_leg + marker_space * 5 + second_leg + third_leg,
                        first_leg + marker_space * 5 + second_leg + third_leg - 10,
                    ],
                    y=[0, 1, 1, 0.5, 0],
                    fill="toself",
                    fillcolor="grey",
                    line_width=0.1,
                    marker=dict(size=0.1),
                    hoveron="fills",
                    text=f"<b>Mode:</b> {scanned_route.loc[2, 'trans_mode']} <br>Price: {scanned_route.loc[2, 'price']}, CO2: {scanned_route.loc[2, 'CO2_eq_kg']}, dist: {scanned_route.loc[2, 'dist_miles']}",
                    hoverinfo="text",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[first_leg + marker_space * 6 + second_leg + third_leg],
                    y=[0.5],
                    marker=dict(size=20, color="black"),
                    text="Arrival",
                    hoverinfo="text",
                )
            )

        elif len(scanned_route) == 1:
            first_leg = 400
            marker_space = 10
            fig = go.Figure(
                go.Scatter(
                    x=[0],
                    y=[0.5],
                    marker=dict(size=20, color="black"),
                    text="Departure",
                    hoverinfo="text",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[
                        marker_space,
                        marker_space,
                        first_leg + marker_space - 10,
                        first_leg + marker_space,
                        first_leg + marker_space - 10,
                    ],
                    y=[0, 1, 1, 0.5, 0],
                    fill="toself",
                    fillcolor="grey",
                    line_width=0.1,
                    marker=dict(size=0.1),
                    hoveron="fills",
                    text=f"Price: {scanned_route.loc[0, 'price']}, CO2: {scanned_route.loc[0, 'CO2_eq_kg']}, dist: {scanned_route.loc[0, 'dist_miles']}",
                    hoverinfo="text",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[first_leg + marker_space * 2],
                    y=[0.5],
                    marker=dict(size=20, color="black"),
                )
            )

        else:
            raise AssertionError(f"not supported route {scanned_route}")

        fig.update_layout(
            height=50,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            yaxis={
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
            },
            xaxis={
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
                "tickfont": {"color": "rgb(128, 128, 128)"},
            },
            margin=dict(l=0, r=0, b=0, t=0),
        )
        return fig

    def scan_route(self, route_detail):
        weights = ["price", "CO2_eq_kg", "duration_h", "dist_miles"]
        new_df = pd.DataFrame(columns=weights)
        new_row = pd.Series(index=weights, data=[0, 0, 0, 0])
        flag_road = 0
        flag_rail = 0
        rail_counter = 0
        for idx, row in route_detail.iterrows():
            if row.trans_mode == "intermodal_link":
                continue
            if row.trans_mode == "road" and flag_road == 0:
                new_row = new_row + row.loc[weights]

            elif row.trans_mode != "road" and flag_road == 0:
                flag_road = 1
                new_row["trans_mode"] = "road"
                new_df = new_df.append(new_row, ignore_index=True)
                new_row = pd.Series(index=weights, data=[0, 0, 0, 0])

            if row.trans_mode == "rail" and flag_rail == 0:
                new_row = new_row + row.loc[weights]
                rail_counter += 1

            elif row.trans_mode != "rail" and flag_rail == 0 and flag_road == 1:
                flag_rail = 1
                new_row["trans_mode"] = "rail"
                new_df = new_df.append(new_row, ignore_index=True)
                new_row = pd.Series(index=weights, data=[0, 0, 0, 0])

            if row.trans_mode == "road" and flag_road == 1:
                new_row = new_row + row.loc[weights]

            if row.trans_mode == "rail" and flag_rail == 1:
                raise AssertionError("Double Rail")

        new_row["trans_mode"] = "road"
        new_df = new_df.append(new_row, ignore_index=True).round(1)
        return new_df, rail_counter

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

    def get_shortest_path(
        self,
        orig: Tuple[float, float],
        dest: Tuple[float, float],
        target_weight: str = "CO2_eq_kg",
        G: Graph = None,
    ):

        # start = time.time()
        if G is None:
            G = self.G_multimodal_u

        node_orig, dist_orig = ox.get_nearest_node(
            G, orig, method="haversine", return_dist=True
        )
        node_dest, dist_dest = ox.get_nearest_node(
            G, dest, method="haversine", return_dist=True
        )

        shortest_path_nodes = nx.astar_path(
            G, node_orig, node_dest, weight=target_weight
        )
        # print(f"get_shortest_path Elapsed time: {time.time() - start}")

        return shortest_path_nodes

    def plot_route(self, path, G=None):

        if G is None:
            G = self.G_multimodal_u

        return folium_revisited.plot_route_folium(
            G,
            path,
            route_width=4,
            route_opacity=0.9,
        )

    def heuristic_func(self, u: int, v: int) -> float:
        return great_circle(
            (self.G_multimodal_u.nodes[u]["y"], self.G_multimodal_u.nodes[u]["x"]),
            (self.G_multimodal_u.nodes[v]["y"], self.G_multimodal_u.nodes[v]["x"]),
        ).miles

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
            self.G_multimodal_u, node_orig, node_dest, weight="dist_miles"
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

    def chosen_path_mode(self, route_details):
        if "intermodal_link" in route_details.index:
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

    def plot_route_summary(self, summary: DataFrame, chosen: str) -> PlotlyFig:

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

        fig = go.Figure()
        s_t = summary[summary.level_1 == "Truckload"]
        s_t_n = norm_summary[norm_summary.level_1 == "Truckload"]
        s_m = summary[summary.level_1 == "Multimodal"]
        s_m_n = norm_summary[norm_summary.level_1 == "Multimodal"]

        if chosen == "Truckload":
            color_t = "rgb(48, 36, 216)"
            color_m = "rgb(170, 167, 209)"
            t_name = "Truck(Chosen)"
            m_name = "Multimodal"
        else:
            color_m = "rgb(48, 36, 216)"
            color_t = "rgb(170, 167, 209)"
            t_name = "Truck"
            m_name = "Multimodal(Chosen)"

        fig.add_trace(
            go.Bar(
                x=s_t.level_0.values,
                y=s_t_n.vals.values,
                textposition="auto",
                text=round(s_t.vals, 1).values,
                orientation="v",
                name=t_name,
                marker_color=color_t,
                hoverinfo='skip'
            )
        )
        fig.add_trace(
            go.Bar(
                x=s_m.level_0.values,
                y=s_m_n.vals.values,
                textposition="auto",
                text=round(s_m.vals, 1).values,
                orientation="v",
                name=m_name,
                marker_color=color_m,
                hoverinfo='skip'
            )
        )

        fig.update_layout(
            # width=600,
            height=180,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        fig.update_layout(
            yaxis={
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
            },
            xaxis={
                "showticklabels": True,
                "showgrid": False,
                "zeroline": False,
                "tickfont": {"color": "rgb(128, 128, 128)"},
            },
            legend=dict(
                orientation="h", font={"color": "rgb(128, 128, 128)"}, y=1.2, x=0.35
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            hoverlabel=None,
        )

        return fig

    def get_terminal_adress(self, path: list) -> list:
        intermodal_terminals_info = pd.read_csv(
            self.script_dir + "/data/intermodal_adress.csv"
        )
        terminal_adresses = []
        for idx in range(len(path)):
            if path[idx] < 200:
                if (
                    self.G_multimodal_u[path[idx - 1]][path[idx]][0]["trans_mode"]
                    == "intermodal_link"
                ):
                    terminal_adresses.append(
                        intermodal_terminals_info.loc[path[idx], "TERM_ADDRE"]
                        + intermodal_terminals_info.loc[path[idx], "TERMINAL"]
                    )
                elif (
                    self.G_multimodal_u[path[idx]][path[idx + 1]][0]["trans_mode"]
                    == "intermodal_link"
                ):
                    terminal_adresses.append(
                        intermodal_terminals_info.loc[path[idx], "TERM_ADDRE"]
                        + intermodal_terminals_info.loc[path[idx], "TERMINAL"]
                    )

        return terminal_adresses

    def rail_route_operators(self, path, rail_edges_counter):
        operators = []
        for u, v in zip(path[:-1], path[1:]):
            d = self.G_multimodal_u[u][v][0]
            for k, v in zip(d.keys(), d.values()):
                if k[:7] == "RROWNER" or k[:7] == "TRKRGHT":
                    operators.append(v)

        main_operators = []
        counter = 0
        for operator, count in Counter(operators).most_common():
            if counter < rail_edges_counter:
                main_operators.append(operator)
                counter += count
            else:
                break

        return main_operators
