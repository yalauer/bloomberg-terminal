"""Composition comparison charts; weights are decimal portfolio fractions."""

import plotly.graph_objects as go

from btlite.viz.theme import apply_theme


def composition_bar(
    data: dict[str, dict[str, float]],
    other: dict[str, dict[str, float]],
    name: str,
    compare: str | None,
    labels: dict[str, str] | None = None,
) -> go.Figure:
    """Show the largest 20 categories; the accompanying table remains complete."""
    categories = sorted(
        data.keys() | other.keys(),
        key=lambda k: -max(sum(data.get(k, {}).values()), sum(other.get(k, {}).values())),
    )[:20]
    fig = go.Figure()
    for title, values in [(name, data), (compare, other)]:
        if title:
            fig.add_trace(
                go.Bar(
                    name=title,
                    y=categories,
                    x=[sum(values.get(k, {}).values()) for k in categories],
                    orientation="h",
                    hovertemplate="%{y}: %{x:.2%}<extra>%{fullData.name}</extra>",
                )
            )
    fig.update_layout(
        barmode="group",
        xaxis_tickformat=".0%",
        yaxis_autorange="reversed",
        height=max(320, len(categories) * 32),
        margin={"l": 160},
    )
    if labels:
        fig.update_yaxes(
            tickmode="array",
            tickvals=categories,
            ticktext=[labels.get(key, key) for key in categories],
        )
    return apply_theme(fig, title="Groesste Anteile am Gesamtdepot (max. 20)")
