import numpy as np
import plotly.graph_objects as go


def plot_spin_evolution(
    spin_evolution,
    steps_to_show: int = 200,
    width: int = 800,
    height: int = 800,
    title: str = "Spin Evolution Heatmap"
):
    """
    Create an interactive heatmap animation of spin evolution with play/pause buttons and a slider.

    Parameters
    ----------
    spin_evolution : np.ndarray or tf.Tensor
        Array of shape (steps, L, L) representing the spin configurations over time.
    steps_to_show : int, optional
        Number of steps to include in the animation (downsamples frames if too many).
    width : int, optional
        Width of the plot in pixels.
    height : int, optional
        Height of the plot in pixels.
    title : str, optional
        Title of the plot.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The interactive Plotly figure object.
    """
    # Ensure numpy
    if hasattr(spin_evolution, "numpy"):
        spin_evolution = spin_evolution.numpy()

    n_steps = spin_evolution.shape[0]
    steps_to_show = min(steps_to_show, n_steps)

    # Select frame indices (evenly spaced)
    frames_idx = np.linspace(0, n_steps - 1, steps_to_show, dtype=int)

    # Create figure with first frame
    fig = go.Figure(
        data=[go.Heatmap(z=spin_evolution[0],
                         colorscale="RdBu", zmin=-1, zmax=1)],
        layout=go.Layout(
            title=title,
            width=width,
            height=height,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False, scaleanchor="x"),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, {"frame": {"duration": 50, "redraw": True},
                                      "fromcurrent": True,
                                      "transition": {"duration": 0}}]),
                    dict(label="Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}])
                ]
            )],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Step: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[str(k)], {"frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate",
                                            "transition": {"duration": 0}}],
                        "label": str(k),
                        "method": "animate"
                    }
                    for k in frames_idx
                ]
            }]
        ),
        frames=[
            go.Frame(
                data=[go.Heatmap(z=spin_evolution[k],
                                 colorscale="RdBu", zmin=-1, zmax=1)],
                name=str(k)
            )
            for k in frames_idx
        ]
    )

    return fig
