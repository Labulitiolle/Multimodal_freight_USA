import re

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
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
Net = MultimodalNet(path_u="mfreight/Multimodal/data/multimodal_G_tot_u.plk")
price_idx = Net.set_price_to_graph()
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
                                        value="(40.439440, -115.057166)",
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
                                        value="(25.382380, -80.475159)",
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
                            options=[
                                {"label": i, "value": i}
                                for i in all_rail_owners
                            ],
                            value=all_rail_owners,
                            multi=True,
                            searchable=True,
                        ),
                    html.Br(),
                    html.Div(
                        className="button-div",
                        children=[html.Button('Update', id='submit-button')]
                    )

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
                                    id="summary-results",
                                    config={"displayModeBar": False,
                                            "responsive":True}
                                )
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="table-container",
                className="table-container",
                children=[
                    html.Div(
                        id="table-upper",
                        children=[
                            html.P("Route summary"),
                            dcc.Loading(children=html.Div(id="stats-container")),
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
                                html.Iframe(id="map", width="100%", height=600),
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
                                        id="route-results",
                                        config={"displayModeBar": False,
                                                "responsive":True},
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
        Output("summary-results", "figure"),
        Output("stats-container", "children"),
        Output("route-results", "figure"),
        Output("rail-operators", "children"),
    ],
    [Input('submit-button', 'n_clicks')],
    state=[
        State("arrival", component_property="value"),
        State("departure", component_property="value"),
        State("operator-selector", component_property="value"),
        State("feature-selector", component_property="value"),
    ],
)
def update_geo_map(n_clicks, select_arrival, select_departure, operators, feature):

    arrival_y = float(re.findall(r"(-?\d+.\d+)\)", select_arrival)[0])
    arrival_x = float(re.findall(r"\((-?\d+.\d+)", select_arrival)[0])

    departure_y = float(re.findall(r"(-?\d+.\d+)\)", select_departure)[0])
    departure_x = float(re.findall(r"\((-?\d+.\d+)", select_departure)[0])

    start = time.time()
    price_target = Net.get_price_target(
        (arrival_x, arrival_y), (departure_x, departure_y), price_idx
    )
    print(f"get_price_target Elapsed time: {time.time() - start}")

    if feature == "Price":
        target = price_target
    else:
        target = feature

    start = time.time()
    removed_edges, removed_nodes = Net.chose_operator_in_graph(operators)
    print(f"chose_operator_in_graph Elapsed time: {time.time() - start}")

    start = time.time()
    path = Net.get_shortest_path(
        (departure_x, departure_y),
        (arrival_x, arrival_y),
        target_weight=target,
    )
    print(f"get_shortest_path Elapsed time: {time.time() - start}")

    start = time.time()
    scanned_route, path_rail_edges = Net.scan_route(
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

    start = time.time()
    chosen_mode = Net.chosen_path_mode(route_details)
    print(f"chosen_path_mode Elapsed time: {time.time() - start}")

    start = time.time()
    table = gen_table(route_details)
    print(f"gen_table Elapsed time: {time.time() - start}")

    start = time.time()
    route_detail_graph = Net.plot_route_detail(scanned_route, path)
    print(f"gen_table Elapsed time: {time.time() - start}")

    start = time.time()
    summary = Net.compute_all_paths(
        departure=(departure_x, departure_y),
        arrival=(arrival_x, arrival_y),
        price_target=price_target,
    )
    print(f"compute_all_paths Elapsed time: {time.time() - start}")

    start = time.time()
    fig_summary = Net.plot_route_summary(summary, chosen_mode)
    print(f"plot_route_summary Elapsed time: {time.time() - start}")

    start = time.time()
    fig = Net.plot_route(path)
    print(f"plot_route Elapsed time: {time.time() - start}")

    start = time.time()
    build_graph.add_nodes_from_df(Net.G_multimodal_u, removed_nodes)
    build_graph.add_edges_from_df(Net.G_multimodal_u, removed_edges)
    print(f"Add Elapsed time: {time.time() - start}")

    return fig._repr_html_(), fig_summary, table, route_detail_graph, operators_string


def gen_table(route_details):

    route_details.reset_index(inplace=True)
    route_detail = route_details.rename(columns={"index": ""})
    return dash_table.DataTable(
        id="stats-table",
        columns=[
            {"name": i, "id": i, "editable": (i == "index")}
            for i in route_detail.columns
        ],
        data=route_details.round(2).to_dict("records"),
        page_size=5,
        style_cell={"background-color": "#242a3b", "color": "#7b7d8d"},
        style_as_list_view=False,
        style_header={"background-color": "#1f2536", "padding": "0px 5px"},
        style_data_conditional=[
            {
                "if": {"column_editable": True},
                "backgroundColor": "#1f2536",
                "color": "#7b7d8d",
            }
        ],
    )


def gen_rail_operators_display(main_operators):
    return r"The rail road displayed is operated by:  " + ", ".join(main_operators)


# Dev
if __name__ == "__main__":
    app.run_server(debug=True)

# server.run(host='0.0.0.0', port=5000)
