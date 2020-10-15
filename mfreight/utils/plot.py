import plotly.graph_objects as go


def make_ternary_plot_grid():
    scale = [i / 5 for i in range(6)]
    data_points = []
    val = 6
    for i in scale:
        for j in [i / (val - 0.99) for i in range(val)]:
            data_points.extend(
                [
                    {
                        "length": round((1 - i) * j, 1),
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
            a=[i for i in map(lambda x: x["length"], data_points)],
            b=[i for i in map(lambda x: x["duration_h"], data_points)],
            c=[i for i in map(lambda x: x["CO2_eq_kg"], data_points)],
            hovertemplate="<extra></extra>"
            + "<br>length: %{a}"
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
                "aaxis": makeAxis("length", 0),
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
