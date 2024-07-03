# /// script
# dependencies = [
#   "plotly",
#   "numpy",
#   "torch",
#
# ]
# ///

from __future__ import annotations

import logging
from typing import Sequence, Union

import plotly.express as px
import plotly.graph_objects as go
from numpy.typing import NDArray
from torch import Tensor


def get_all_named_cmaps() -> list[str]:
    named_colorscales = px.colors.named_colorscales()
    base_cmaps = ["cividis", "sunset", "turbo", "thermal"]
    base_cmaps_idx = [named_colorscales.index(x) for x in base_cmaps]
    for idx in base_cmaps_idx:
        named_colorscales.pop(idx)
    named_colorscales = base_cmaps + named_colorscales
    return named_colorscales


def interactive_plot(
    data: Union[NDArray, Sequence[NDArray]],
    labels: Union[str, Sequence[str]] = None,
    point_size: int = 3,
    opacity: float = 0.8,
    colorbar: bool = False,
    color: NDArray = None,
    color_range: tuple[int] = None,
    cmap: str = None,
    constraint_x: bool = False,
    constraint_y: bool = False,
    constraint_z: bool = False,
    return_fig: bool = False,
    width: int = 500,
    height: int = 500,
    title: str = None,
):
    """Interactive plot of point cloud(s) based on Plotly. Can display N pointcloud(s).

    Args:
        pointcloud (Union[np.ndarray, tuple[np.ndarray]]): If a list or tuple is passed,
            each element will be displayed with its own name, colormap, and button to
            toggle visibility.
        label (str, optional): This will be the title of the plot. Is is called label
            because it was intended to be used within a classification setup.
            Defaults to None.
        point_size (int, optional): Display size of one point (x, y, z). Defaults to 1.
        color (np.ndarray, optional): Color of each points. MUST be a sequence of length
            equals to the number of points.
            If None, the z coordinates will be used to color points.
            Defaults to None.
        constraint_x (bool, optional): Rescale x within [-1 ,1]. Defaults to False.
        constraint_y (bool, optional): Rescale y within [-1 ,1]. Defaults to False.
        constraint_z (bool, optional): Rescale z within [-1 ,1]. Defaults to False.

    Raises:
        ValueError: If too few or too many pointclouds are passed to the function.
    """
    if not isinstance(data, (list, tuple)):
        data = [data]
    if isinstance(data, tuple):
        data = list(data)
    # move to cpu if required; raises warning
    moved_to_cpu = False
    for i, x in enumerate(data):
        if isinstance(x, Tensor) and x.is_cuda:
            data[i] = x.cpu()
            moved_to_cpu = True
    if moved_to_cpu:
        logging.basicConfig(format="[%(levelname)s] %(message)s")
        logging.warning(
            (
                "CUDA Tensors were passed in: "
                "I had to move them to the cpu in order to display them."
            )
        )
    N = len(data)
    if labels is None:
        labels = [f"pointcloud {i + 1}" for i in range(N)]
    if not isinstance(labels, (list, tuple)):
        labels = (labels,)
    if labels is not None and not len(data) == len(labels):
        raise ValueError(f"You gave {len(data)} pointclouds but {len(labels)} labels.")
    all_cmaps = get_all_named_cmaps()
    if isinstance(cmap, str):
        cmaps = N * [cmap]
    elif not (isinstance(cmap, (tuple, list)) and len(cmap) == len(data)):
        cmaps = all_cmaps[:N]
    else:
        cmaps = cmap
    traces = list()
    for pointcloud, label, cmap in zip(data, labels, cmaps):
        x, y, z = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]
        c = color if color is not None else z
        marker_kwargs = dict(size=point_size, opacity=opacity, color=c, colorscale=cmap)
        if color_range is not None:
            marker_kwargs["cmin"] = color_range[0]
            marker_kwargs["cmax"] = color_range[1]
        if colorbar:
            marker_kwargs["colorbar"] = dict(thickness=20)
        scatter_kwargs = dict(
            visible=True, mode="markers", name=label, marker=marker_kwargs
        )
        traces.append(go.Scatter3d(x=x, y=y, z=z, **scatter_kwargs))
    layout = dict(
        width=width,
        height=height,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        margin=dict(t=50),
    )
    layout["scene"] = dict()
    if constraint_x:
        layout["scene"]["xaxis"] = dict(nticks=4, range=[-1, 1])
    if constraint_y:
        layout["scene"]["yaxis"] = dict(nticks=4, range=[-1, 1])
    if constraint_z:
        layout["scene"]["zaxis"] = dict(nticks=4, range=[-1, 1])
    if title is not None:
        layout["title"] = title
    fig = go.Figure(data=traces, layout=layout)
    if return_fig:
        return fig
    fig.show()
