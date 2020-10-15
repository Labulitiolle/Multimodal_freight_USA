import re

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
from mfreight.Multimodal.graph_utils import MultimodalNet
from mfreight.utils.plot import make_ternary_selector

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
                            value=[],
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

Net = MultimodalNet()


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

    ctx = dash.callback_context
    if ctx.triggered[0]["prop_id"].split(".")[0] == "operator-select-all":
        if select_all == ["All"]:
            value = [i["value"] for i in options]
        else:
            value = all_rail_owners[:4]
    else:
        value = all_rail_owners[:4]

    Net.chose_operator(value)
    return value, options, "Multimodal route"


@app.callback(
    Output("map", "srcDoc"),
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

    Net.chose_operator(operators)
    Net.set_taget_weight_to_graph(weights["a"], weights["b"], weights["c"])
    fig = Net.plot_route(
        (departure_x, departure_y),
        (arrival_x, arrival_y),
        target_weight="target_feature",
        folium=True,
    )
    return fig._repr_html_()


@app.callback(
    Output("stats-container", "children"),
    [
        Input("arrival", component_property="value"),
        Input("departure", component_property="value"),
        Input("operator-selector", component_property="value"),
        Input("feature-selector", component_property="clickData"),
    ],
)
def update_route_datatable(select_arrival, select_departure, operators, weights):
    if weights:
        weights = weights["points"][0]
    else:
        weights = {"a": 0, "b": 0, "c": 1}

    arrival_y = float(re.findall(r"(-?\d+.\d+)\)", select_arrival)[0])
    arrival_x = float(re.findall(r"\((-?\d+.\d+)", select_arrival)[0])

    departure_y = float(re.findall(r"(-?\d+.\d+)\)", select_departure)[0])
    departure_x = float(re.findall(r"\((-?\d+.\d+)", select_departure)[0])

    Net.chose_operator(operators)
    Net.set_taget_weight_to_graph(weights["a"], weights["b"], weights["c"])
    df = Net.get_route_detail(
        (departure_x, departure_y), (arrival_x, arrival_y), "target_feature"
    )

    df.reset_index(inplace=True)
    df = df.rename(columns={"index": ""})
    return dash_table.DataTable(
        id="stats-table",
        columns=[{"name": i, "id": i, "editable": (i == "index")} for i in df.columns],
        data=df.round(2).to_dict("rows"),
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


if __name__ == "__main__":
    app.run_server(debug=True)
