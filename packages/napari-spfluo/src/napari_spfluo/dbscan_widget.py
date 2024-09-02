import napari
import numpy as np
from magicgui import magic_factory
from napari.layers import Labels
from sklearn.cluster import DBSCAN, HDBSCAN


@magic_factory(
    epsilon={
        "widget_type": "FloatSlider",
        "min": 0.1,
        "max": 10,
    },
    min_samples={
        "widget_type": "IntSlider",
        "min": 1,
        "max": 500,
    },
)
def run_dbscan(
    labels: "napari.layers.Labels",
    epsilon: "float" = 0.5,
    min_samples: "int" = 100,
) -> "napari.layers.Labels":
    points = np.stack(np.nonzero(labels.data), axis=-1)
    dbscan = DBSCAN(eps=epsilon, min_samples=int(min_samples))
    dbscan.fit(points)
    dbscan_labels = Labels(
        data=np.zeros(labels.data.shape, dtype=int),
        name=labels.name + " DBSCAN",
        scale=labels.scale,
    )
    for x in range(-1, dbscan.labels_.max() + 1):
        dbscan_labels.data[tuple(points[dbscan.labels_ == x].T)] = x + 1
    return dbscan_labels


@magic_factory(
    min_cluster_size={
        "widget_type": "IntSlider",
        "min": 0,
        "max": 1000,
    },
    cluster_selection_epsilon={
        "widget_type": "IntSlider",
        "min": 0,
        "max": 10,
    },
)
def run_hdbscan(
    labels: "napari.layers.Labels",
    min_cluster_size: "int" = 5,
    cluster_selection_epsilon: "int" = 0,
) -> "napari.layers.Labels":
    points = np.stack(np.nonzero(labels.data), axis=-1)
    dbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    dbscan.fit(points)
    dbscan_labels = Labels(
        data=np.zeros(labels.data.shape, dtype=int),
        name=labels.name + " HDBSCAN",
        scale=labels.scale,
    )
    for x in range(-1, dbscan.labels_.max() + 1):
        dbscan_labels.data[tuple(points[dbscan.labels_ == x].T)] = x + 1
    return dbscan_labels
