import plotly.graph_objects as go


def plot_route_detail(scanned_route, terminal_adresses):

    scanned_route_norm = scanned_route.loc[
        :, ["price", "CO2_eq_kg", "duration_h", "dist_miles"]
    ] / scanned_route.loc[:, ["price", "CO2_eq_kg", "duration_h", "dist_miles"]].sum(
        axis=0
    )

    scatter_color='#611705'
    road_color='grey'

    if len(scanned_route) == 3:
        first_leg = 400 * scanned_route_norm.loc[0, "dist_miles"]
        second_leg = 400 * scanned_route_norm.loc[1, "dist_miles"]
        third_leg = 400 * scanned_route_norm.loc[2, "dist_miles"]

        # Make sure that that the polygon will have a  decent minimal size
        if third_leg < 12:
            third_leg = 12
        if first_leg < 12:
            first_leg = 12

        marker_space = 10
        fig = go.Figure(
            go.Scatter(
                x=[0],
                y=[0.5],
                marker=dict(size=20, color=scatter_color, symbol='x'),
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
                fillcolor=road_color,
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
                marker=dict(size=20, color=scatter_color),
                text=f"<b>Terminal:</b> <br>{terminal_adresses[0]}",
                hoverinfo="text",
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
                fillcolor="#008000",
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
                marker=dict(size=20, color=scatter_color),
                text=f"<b>Terminal:</b> <br>{terminal_adresses[1]}",
                hoverinfo="text",
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
                fillcolor=road_color,
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
                marker=dict(size=20, color=scatter_color, symbol='star'),
                text="Arrival",
                hoverinfo="text",
            )
        )

    elif len(scanned_route) == 2:
        first_leg = 400 * scanned_route_norm.loc[0, "dist_miles"]
        second_leg = 400 * scanned_route_norm.loc[1, "dist_miles"]

        if scanned_route.loc[0, "trans_mode"] == "road":
            first_leg_color = road_color
            second_leg_color = "#008000"
        else:
            first_leg_color = "#008000"
            second_leg_color = road_color

        if first_leg < 12:
            first_leg = 12
        if second_leg < 12:
            second_leg = 12

        marker_space = 10
        fig = go.Figure(
            go.Scatter(
                x=[0],
                y=[0.5],
                marker=dict(size=20, color=scatter_color, symbol='x'),
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
                fillcolor=first_leg_color,
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
                marker=dict(size=20, color=scatter_color),
                text=f"<b>Terminal:</b> <br>{terminal_adresses[0]}",
                hoverinfo="text",
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
                fillcolor=second_leg_color,
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
                marker=dict(size=20, color=scatter_color, symbol='star'),
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
                marker=dict(size=20, color=scatter_color, symbol='x'),
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
                fillcolor=road_color,
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
                marker=dict(size=20, color=scatter_color),
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

def bar_plot_summary(summary, norm_summary, chosen_mode):
    fig = go.Figure()
    s_t = summary[summary.level_1 == "Truckload"]
    s_t_n = norm_summary[norm_summary.level_1 == "Truckload"]
    s_m = summary[summary.level_1 == "Multimodal"]
    s_m_n = norm_summary[norm_summary.level_1 == "Multimodal"]

    if chosen_mode == "Truckload":
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
            hoverinfo="skip",
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
            hoverinfo="skip",
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




