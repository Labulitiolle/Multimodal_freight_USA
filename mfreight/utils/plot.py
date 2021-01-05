import plotly.graph_objects as go
from mfreight.utils import build_graph
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon


def make_ternary_plot_grid():
    scale = [i / 5 for i in range(6)]
    data_points = []
    val = 6
    for i in scale:
        for j in [i / (val - 0.99) for i in range(val)]:
            data_points.extend(
                [
                    {
                        "price": round((1 - i) * j, 1),
                        "duration_h": round(i, 1),
                        "CO2_eq_kg": round((1 - i) * (1 - j), 1),
                    }
                ]
            )
        val = val - 1
    return data_points


def makeAxis(title: str, tickangle: int):
    return {
        "title": title,
        "titlefont": {"size": 10, "color": "rgb(255, 255, 255)"},
        "tickangle": tickangle,
        "tickfont": {"size": 7, "color": "rgb(128, 128, 128)"},
        "ticklen": 5,
        "showline": True,
        "showgrid": True,
    }


def make_ternary_selector():
    data_points = make_ternary_plot_grid()
    fig = go.Figure(
        go.Scatterternary(
            mode="markers",
            a=[i for i in map(lambda x: x["price"], data_points)],
            b=[i for i in map(lambda x: x["duration_h"], data_points)],
            c=[i for i in map(lambda x: x["CO2_eq_kg"], data_points)],
            hovertemplate="<extra></extra>"
            + "<br>price: %{a}"
            + "<br>duration: %{b}"
            + "<br>CO2: %{c}",
            showlegend=False,
            marker={
                "size": 2,
            },
        )
    )

    fig.update_layout(
        {
            "ternary": {
                "sum": 1,
                "aaxis": makeAxis("price", 0),
                "baxis": makeAxis("<br>duration", 0),
                "caxis": makeAxis("<br>CO2", 0),
            },
        }
    )
    fig.update_layout(
        width=300,
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_graph(
    G,
    ax=None,
    figsize=(8, 8),
    bgcolor="#111111",
    node_color="w",
    node_size=15,
    node_alpha=None,
    node_edgecolor="none",
    node_zorder=1,
    edge_color="#999999",
    edge_linewidth=1,
    edge_alpha=None,
    show=True,
    bbox=None,
):
    """
    Plot a graph.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    ax : matplotlib axis
        if not None, plot on this preexisting axis
    figsize : tuple
        if ax is None, create new figure with size (width, height)
    bgcolor : string
        background color of plot
    node_color : string or list
        color(s) of the nodes
    node_size : int
        size of the nodes: if 0, then skip plotting the nodes
    node_alpha : float
        opacity of the nodes, note: if you passed RGBA values to node_color,
        set node_alpha=None to use the alpha channel in node_color
    node_edgecolor : string
        color of the nodes' markers' borders
    node_zorder : int
        zorder to plot nodes: edges are always 1, so set node_zorder=0 to plot
        nodes below edges
    edge_color : string or list
        color(s) of the edges' lines
    edge_linewidth : float
        width of the edges' lines: if 0, then skip plotting the edges
    edge_alpha : float
        opacity of the edges, note: if you passed RGBA values to edge_color,
        set edge_alpha=None to use the alpha channel in edge_color
    show : bool
        if True, call pyplot.show() to show the figure
    close : bool
        if True, call pyplot.close() to close the figure
    save : bool
        if True, save the figure to disk at filepath
    filepath : string
        if save is True, the path to the file. file format determined from
        extension. if None, use settings.imgs_folder/image.png
    dpi : int
        if save is True, the resolution of saved file
    bbox : tuple
        bounding box as (north, south, east, west). if None, will calculate
        from spatial extents of plotted geometries.
    Returns
    -------
    fig, ax : tuple
        matplotlib figure, axis
    """
    max_node_size = max(node_size) if hasattr(node_size, "__iter__") else node_size
    max_edge_lw = (
        max(edge_linewidth) if hasattr(edge_linewidth, "__iter__") else edge_linewidth
    )
    if max_node_size <= 0 and max_edge_lw <= 0:
        raise ValueError(
            "Either node_size or edge_linewidth must be > 0 to plot something."
        )

    # create fig, ax as needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=bgcolor, frameon=False)
        ax.set_facecolor(bgcolor)
    else:
        fig = ax.figure

    if max_edge_lw > 0:
        # plot the edges' geometries
        gdf_edges = build_graph.graph_to_gdfs2(G, nodes=False)["geometry"]
        ax = gdf_edges.plot(
            ax=ax, color=edge_color, lw=edge_linewidth, alpha=edge_alpha, zorder=1
        )

    if max_node_size > 0:
        # scatter plot the nodes' x/y coordinates
        gdf_nodes = build_graph.graph_to_gdfs2(G, edges=False, node_geometry=False)[
            ["x", "y"]
        ]
        ax.scatter(
            x=gdf_nodes["x"],
            y=gdf_nodes["y"],
            s=node_size,
            c=node_color,
            alpha=node_alpha,
            edgecolor=node_edgecolor,
            zorder=node_zorder,
        )

    # get spatial extents from bbox parameter or the edges' geometries
    padding = 0
    if bbox is None:
        try:
            west, south, east, north = gdf_edges.total_bounds
        except NameError:
            west, south = gdf_nodes.min()
            east, north = gdf_nodes.max()
        bbox = north, south, east, west
        padding = 0.02  # pad 2% to not cut off peripheral nodes' circles
        ax = _config_ax(ax, G.graph["crs"], bbox, padding)
    else:
        ax = _config_ax(ax, G.graph["crs"], bbox, padding)

    # configure axis appearance, save/show figure as specified, and return

    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, ax


def _config_ax(ax, crs, bbox, padding):
    """
    Configure axis for display.
    Parameters
    ----------
    ax : matplotlib axis
        the axis containing the plot
    crs : dict or string or pyproj.CRS
        the CRS of the plotted geometries
    bbox : tuple
        bounding box as (north, south, east, west)
    padding : float
        relative padding to add around the plot's bbox
    Returns
    -------
    ax : matplotlib axis
        the configured/styled axis
    """
    # set the axis view limits to bbox + relative padding
    north, south, east, west = bbox
    padding_ns = (north - south) * padding
    padding_ew = (east - west) * padding
    ax.set_ylim((south - padding_ns, north + padding_ns))
    ax.set_xlim((west - padding_ew, east + padding_ew))

    # set margins to zero, point ticks inward, turn off ax border and x/y axis
    # so there is no space around the plot
    ax.margins(0)
    ax.tick_params(which="both", direction="in")
    _ = [s.set_visible(False) for s in ax.spines.values()]
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # set aspect ratio

    # if data are not projected, conform aspect ratio to not stretch plot
    coslat = np.cos((south + north) / 2.0 / 180.0 * np.pi)
    ax.set_aspect(1.0 / coslat)
    # else:
    #     # if projected, make everything square
    #     ax.set_aspect("equal")

    return ax


def plot_usa_background(bbox, ax=None):
    bbox_polygon = Polygon(
        [(bbox[3], bbox[1]), (bbox[3], bbox[0]), (bbox[2], bbox[0]), (bbox[2], bbox[1])]
    )
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    n_america = world[world["name"] == "United States of America"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))

    n_america_clipped = gpd.clip(n_america, bbox_polygon)
    ax = n_america_clipped.plot(ax=ax, edgecolor="#D8D8D8", color="#D8D8D8", alpha=0.5)
    plt.axis("off")

    return ax


def plot_multimodal_graph(
    G,
    bbox=None,
    show=True,
    save_path=None,
    background=False,
    ax=None,
    show_intermodal=True,
    rail_width=1,
    rail_alpha=0.5,
    road_width=1,
    road_alpha=0.1,
    res=150,
    title=None,
):
    """
    :param bbox: (north,south,east,west)

    :return: axis
    """

    nodes_road, nodes_rail = [], []
    nodes_intermodal_geometry = []
    if bbox:
        for n, data in G.nodes(data=True):
            if bbox[3] < data["x"] < bbox[2] and bbox[1] < data["y"] < bbox[0]:
                if data["trans_mode"] == "road":
                    nodes_road.append(n)
                elif data["trans_mode"] == "rail" or data["trans_mode"] == "intermodal":
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

    if bbox is None:
        bbox = (52, 22, -66, -126)

    G_road = G.subgraph(nodes_road)
    G_rail = G.subgraph(nodes_rail)
    intermodal_nodes = gpd.GeoDataFrame(
        {"geometry": nodes_intermodal_geometry}, crs="epsg:4326"
    )

    if background:
        ax = plot_usa_background(bbox=bbox, ax=ax)

    else:
        fig, ax = plt.subplots(figsize=(20, 10))
        print('shiit')

    if title:
        ax.set_title(title, color="grey", fontsize=24)

    if len(G_rail.edges()) > 1:

        fig, ax = plot_graph(
            G_rail,
            edge_color="green",
            edge_linewidth=rail_width,
            edge_alpha=rail_alpha,
            node_color="yellow",
            node_size=1,
            node_alpha=0.1,
            bgcolor="white",
            show=False,
            ax=ax,
            bbox=bbox,
        )

        if show_intermodal:
            ax = intermodal_nodes.plot(ax=ax, color="red", alpha=1, markersize=2)

    if len(G_road.edges()) > 1:
        fig, ax = plot_graph(
            G_road,
            edge_color="blue",
            edge_linewidth=road_width,
            edge_alpha=road_alpha,
            node_color="blue",
            bgcolor="white",
            node_size=0.01,
            node_alpha=0.2,
            show=False,
            ax=ax,
            bbox=bbox,
        )

    if save_path:
        plt.savefig(save_path, dpi=res, pad_inches=0, bbox_inches='tight')
    if show:
        plt.show()
        # CLOSE PLOTS
    plt.close("all")
    return ax


def plot_multimodal_path_search(G, orig, dest, save_path, show, res=150, title=None):
    fig, ax = plt.subplots(figsize=(14, 7))
    orig_point = gpd.GeoDataFrame({"geometry": [orig]}, crs="epsg:4326")

    dest_point = gpd.GeoDataFrame({"geometry": [dest]}, crs="epsg:4326")
    orig_point.plot(ax=ax, color="brown", alpha=1, markersize=45, marker="x")
    dest_point.plot(ax=ax, color="brown", alpha=1, markersize=45, marker="*")

    ax = plot_multimodal_graph(
        G,
        bbox=None,
        show=show,
        save_path=save_path,
        background=True,
        ax=ax,
        show_intermodal=False,
        res=res,
        title=title,
    )


def plot_multimodal_route(G, orig, dest, save_path, route, show, res=150, title=None):
    fig, ax = plt.subplots(figsize=(14, 7))
    orig_point = gpd.GeoDataFrame({"geometry": [orig]}, crs="epsg:4326")

    dest_point = gpd.GeoDataFrame({"geometry": [dest]}, crs="epsg:4326")
    orig_point.plot(ax=ax, color="brown", alpha=1, markersize=45, marker="x")
    dest_point.plot(ax=ax, color="brown", alpha=1, markersize=45, marker="*")
    return plot_multimodal_graph(
        G.subgraph(route),
        bbox=None,
        show=show,
        save_path=save_path,
        background=True,
        ax=ax,
        show_intermodal=False,
        rail_width=2,
        rail_alpha=0.8,
        road_width=2,
        road_alpha=0.8,
        res=res,
        title=title,
    )
