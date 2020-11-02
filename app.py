import re

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
from mfreight.Multimodal.graph_utils import MultimodalNet
from mfreight.utils.plot import make_ternary_selector
from mfreight.utils import build_graph

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

fig = make_ternary_selector()


def build_upper_left_panel():
    return html.Div(
        id="upper-left",
        className="six columns",
        children=[
            html.P(
                className="section-title",
                children="Chose route",
            ),
            html.Div(
                className="control-row-1",
                children=[
                    html.Div(
                        id="select-departure_id",
                        children=[
                            html.Label("Enter a (lat, long) departure position"),
                            dcc.Input(
                                id="departure",
                                value="(30.439440, -85.057166)",
                                type="text",
                            ),
                        ],
                    ),
                    html.Div(
                        id="select-arrival_id",
                        children=[
                            html.Label("Enter a (lat, long) destination position"),
                            dcc.Input(
                                id="arrival",
                                value="(25.382380, -80.475159)",
                                type="text",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="feature-selector-outer",
                className="control-row-2",
                children=[
                    html.Label("Select target feature"),
                    html.Div(
                        id="graph-container",
                        children=dcc.Graph(
                            id="feature-selector",
                            figure=fig,
                        ),
                    ),
                ],
            ),
            html.Div(
                id="operator-selector-outer",
                className="control-row-3",
                children=[
                    html.Label("Select rail operators"),
                    html.Div(
                        id="checklist-container",
                        children=dcc.Checklist(
                            id="operator-select-all",
                            options=[{"label": "Select All Operators", "value": "All"}],
                            value=["All"],
                        ),
                    ),
                    html.Div(
                        id="operator-select-dropdown-outer",
                        children=dcc.Dropdown(
                            id="operator-selector",
                            multi=True,
                            searchable=True,
                        ),
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
                    ],
                ),
            ],
        ),
    ],
)

Net = MultimodalNet(path_u= "mfreight/Multimodal/data/multimodal_G_tot_u.plk")
price_idx = Net.set_price_to_graph()


@app.callback(
    [
        Output("operator-selector", "value"),
        Output("operator-selector", "options"),
        Output("map-title", "children"),
    ],
    [
        Input("operator-select-all", "value"),
    ],
)
def update_operator_dropdown(select_all):
    all_rail_owners = Net.get_rail_owners()

    options = [{"label": i, "value": i} for i in all_rail_owners]

    if select_all == ["All"]:
        value = [i["value"] for i in options]
    else:
        value = all_rail_owners[:4]

    return value, options, "Multimodal route"


@app.callback(
    [Output("map", component_property="srcDoc"), Output("stats-container", "children")],
    [
        Input("arrival", component_property="value"),
        Input("departure", component_property="value"),
        Input("operator-selector", component_property="value"),
        Input("feature-selector", component_property="clickData"),
    ],
)
def update_geo_map(select_arrival, select_departure, operators, weights):
    if weights:
        weights = weights["points"][0]
    else:
        weights = {"a": 0, "b": 0, "c": 1}

    arrival_y = float(re.findall(r"(-?\d+.\d+)\)", select_arrival)[0])
    arrival_x = float(re.findall(r"\((-?\d+.\d+)", select_arrival)[0])

    departure_y = float(re.findall(r"(-?\d+.\d+)\)", select_departure)[0])
    departure_x = float(re.findall(r"\((-?\d+.\d+)", select_departure)[0])

    orig_state = Net.extract_state((departure_x, departure_y))
    dest_state = Net.extract_state((arrival_x, arrival_y))

    if (orig_state, dest_state) in price_idx:
        price_target = (orig_state, dest_state)
    else:
        price_target = ('default', 'default')

    if weights["a"] > weights["b"] and weights["a"] > weights["c"]:
        target = price_target
    elif weights["b"] > weights["a"] and weights["b"] > weights["c"]:
        target = 'duration_h'
    elif weights["c"] > weights["a"] and weights["c"] > weights["b"]:
        target = 'CO2_eq_kg'
    else:
        target = 'price' # TODO simplify feature selector
    removed_edges, removed_nodes = Net.chose_operator_in_graph(operators)

    fig, path = Net.plot_route(
        (departure_x, departure_y),
        (arrival_x, arrival_y),
        target_weight=target,
        folium=True,
    )

    table = gen_table(Net.route_detail_from_graph(path, price_target=price_target))
    build_graph.add_nodes_from_df(Net.G_multimodal_u, removed_nodes)
    build_graph.add_edges_from_df(Net.G_multimodal_u, removed_edges)


    return fig._repr_html_(), table


def gen_table(route_detail):

    route_detail.reset_index(inplace=True)
    route_detail = route_detail.rename(columns={"index": ""})
    return dash_table.DataTable(
        id="stats-table",
        columns=[
            {"name": i, "id": i, "editable": (i == "index")}
            for i in route_detail.columns
        ],
        data=route_detail.round(2).to_dict("rows"),
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

# Dev
if __name__ == "__main__":
    app.run_server(debug=True)

# server.run(host='0.0.0.0', port=5000)
