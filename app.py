import re

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from mfreight.Multimodal.graph_utils import MultimodalNet
from mfreight.utils import build_graph
import time

app = dash.Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no",
        }
    ],
)
server = app.server


start = time.time()
Net = MultimodalNet(path_u="mfreight/Multimodal/data/multimodal_G_tot_u_w_price.plk")
print(f"Loaded in Elapsed time: {time.time() - start}")

all_rail_owners = Net.get_rail_owners()
options = [{"label": i, "value": i} for i in all_rail_owners]
print(f"Refeshed in Elapsed time: {time.time() - start}")


def build_upper_left_panel():
    return html.Div(
        [
            html.Div(
                children=[
                    html.P(
                        className="control_label",
                        children="Chose route",
                    ),
                    html.Div(
                        className="control-row-1",
                        children=[
                            html.Div(
                                id="select-departure_id",
                                children=[
                                    html.Label(
                                        "Enter a (lat, long) departure position"
                                    ),
                                    dcc.Input(
                                        id="departure",
                                        value="(27.938220, -81.698181)",#(34.050717, -118.288621)
                                        type="text",
                                    ),
                                ],
                            ),
                            html.Div(
                                id="select-arrival_id",
                                children=[
                                    html.Label(
                                        "Enter a (lat, long) destination position"
                                    ),
                                    dcc.Input(
                                        id="arrival",
                                        value="(41.815994, -87.670207)",
                                        type="text",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Br(),
                    html.P(
                        className="control_label",
                        children="Select target feature",
                    ),
                    html.Br(),
                    dcc.RadioItems(
                        id="feature-selector",
                        options=[
                            {"label": "CO2 Emissions", "value": "CO2_eq_kg"},
                            {"label": "Price", "value": "Price"},
                            {"label": "Duration", "value": "duration_h"},
                        ],
                        value="CO2_eq_kg",
                    ),
                    html.Br(),
                    html.P(
                        className="control_label",
                        children="Select rail operators",
                    ),
                    html.Br(),
                    dcc.Dropdown(
                        id="operator-selector",
                        options=[{"label": i, "value": i} for i in all_rail_owners],
                        value=all_rail_owners,
                        multi=True,
                        searchable=True,
                    ),
                    html.Br(),
                    html.Div(
                        className="button-div",
                        children=[html.Button("Update", id="submit-button")],
                    ),
                ],
                className="pretty_container",
                id="upper-left-param",
            ),
            html.Div(
                id="graph-container",
                className="graph-container",
                children=[
                    html.Div(
                        id="graph-upper",
                        children=[
                            html.P("Route results"),
                            dcc.Loading(
                                children=dcc.Graph(
                                    id="bar-plot-results",
                                    config={
                                        "displayModeBar": False,
                                        "responsive": True,
                                    },
                                )
                            ),
                        ],
                    ),
                ],
            ),
        ],
        id="upper-left",
        className="six columns",
    )


app.layout = html.Div(
    className="container scalable",
    children=[
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.H6("Intermodal freight network"),
                html.Img(src=app.get_asset_url("logo.png")),
            ],
        ),
        html.Div(
            id="upper-container",
            className="row",
            children=[
                build_upper_left_panel(),
                html.Div(
                    id="geo-map-outer",
                    className="six columns",
                    children=[
                        html.P(
                            id="map-title",
                            children="Geomap",
                        ),
                        html.Div(
                            id="geo-map-loading-outer",
                            children=[
                                html.H1("Multimodal optimized route"),
                                dcc.Loading(
                                children=html.Iframe(id="map", width="100%", height=600))
                            ],
                        ),
                        html.Div(
                            id="graph-right",
                            children=[
                                html.P(
                                    className="section-title",
                                    children="Route details",
                                ),
                                dcc.Loading(
                                    children=dcc.Graph(
                                        id="route-visual-summary",
                                        config={
                                            "displayModeBar": False,
                                            "responsive": True,
                                        },
                                    )
                                ),
                            ],
                        ),
                        html.Br(),
                        html.Div(className="explanatory-text", id="rail-operators"),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    [
        Output("map", component_property="srcDoc"),
        Output("bar-plot-results", "figure"),
        Output("route-visual-summary", "figure"),
        Output("rail-operators", "children"),
    ],
    [Input("submit-button", "n_clicks")],
    state=[
        State("arrival", component_property="value"),
        State("departure", component_property="value"),
        State("operator-selector", component_property="value"),
        State("feature-selector", component_property="value"),
    ],
)
def update_geo_map(n_clicks, select_arrival, select_departure, operators, feature):

    departure = format_input_positions(select_departure)
    arrival = format_input_positions(select_arrival)

    start = time.time()
    price_target = Net.get_price_target(departure, arrival)
    print(f"get_price_target Elapsed time: {time.time() - start}")

    start = time.time()
    removed_edges, removed_nodes = Net.chose_operator_in_graph(operators)
    print(f"chose_operator_in_graph Elapsed time: {time.time() - start}")

    start = time.time()
    path = Net.get_shortest_path(
        departure,
        arrival,
        target_weight=feature,
        price_target=price_target
    )
    print(f"get_shortest_path Elapsed time: {time.time() - start}")

    start = time.time()
    scanned_route, path_rail_edges, stop_list = Net.scan_route(
        Net.route_detail_from_graph(
            path, show_entire_route=True, price_target=price_target
        )
    )
    print(f"scan_route Elapsed time: {time.time() - start}")

    start = time.time()
    operators_string = gen_rail_operators_display(
        Net.rail_route_operators(path, path_rail_edges)
    )
    print(f"gen_rail_operators_display Elapsed time: {time.time() - start}")

    start = time.time()
    route_details = Net.route_detail_from_graph(path, price_target=price_target)
    print(f"route_detail_from_graph Elapsed time: {time.time() - start}")

    chosen_mode = Net.chosen_path_mode(route_details)

    start = time.time()
    route_visual_summary = Net.plot_route_visual_summary(scanned_route, path)
    print(f"gen_table Elapsed time: {time.time() - start}")

    start = time.time()
    route_specs = Net.compute_all_paths(
        departure=departure,
        arrival=arrival,
        price_target=price_target,
    )
    print(f"compute_all_paths Elapsed time: {time.time() - start}")

    start = time.time()
    bar_plot_results = Net.generate_bar_plot(route_specs, chosen_mode)
    print(f"plot_route_summary Elapsed time: {time.time() - start}")

    start = time.time()
    fig = Net.plot_route(path, stop_list)
    print(f"plot_route Elapsed time: {time.time() - start}")

    start = time.time()
    build_graph.add_nodes_from_df(Net.G_multimodal_u, removed_nodes)
    build_graph.add_edges_from_df(Net.G_multimodal_u, removed_edges)
    print(f"Add Elapsed time: {time.time() - start}")

    return fig._repr_html_(), bar_plot_results, route_visual_summary, operators_string


def gen_rail_operators_display(main_operators):
    return r"The rail road displayed is operated by:  " + ", ".join(main_operators)

def format_input_positions(input_string):
    position_y = float(re.findall(r"(-?\d+.\d+)\)", input_string)[0])
    position_x = float(re.findall(r"\((-?\d+.\d+)", input_string)[0])
    return (position_x, position_y)

# Dev
if __name__ == "__main__":
    app.run_server(debug=True)

# server.run(host='0.0.0.0', port=5000)
