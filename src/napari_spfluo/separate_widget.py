from typing import List  # noqa: UP035

import napari
import numpy as np
from magicgui import magic_factory
from napari.layers import Image
from spfluo.utils.separate import _separate_clusters


@magic_factory(
    tukey_alpha={
        "widget_type": "FloatSlider",
        "min": 0,
        "max": 1,
    },
)
def run_separate(
    image: "napari.layers.Image",
    clusters: "napari.layers.Labels",
    tukey_alpha: float = 0.1,
) -> "List[napari.layers.Layer]":
    image: Image = image
    image_data: np.ndarray = image.data
    clusters_data: np.ndarray = clusters.data
    scale = image.scale
    assert not (image_data.ndim < 3 or image_data.ndim > 4)
    assert (
        clusters_data.ndim == image_data.ndim
        and clusters_data.shape == image_data.shape
    )
    if image_data.ndim == 4:
        image_data = image_data[0]
        clusters_data = clusters_data[0]
        scale = scale[1:]
    clusters_data[clusters_data < 0] = 0
    labels = np.nonzero(np.bincount(clusters_data.flatten()))[0]
    assert len(labels) == 3, "There should be only 2 clusters_data"
    clusters_coords = np.nonzero(clusters_data)
    clusters_labels = np.empty(clusters_coords[0].shape, dtype=int)
    for label, new_label in zip((labels[1], labels[2]), (0, 1)):
        clusters_labels[clusters_data[clusters_coords] == label] = new_label
    (im1, im2), centers = _separate_clusters(
        image_data[None],
        np.stack(clusters_coords, axis=-1),
        clusters_labels,
        image_data[clusters_coords],
        image_data.shape,
        (1, 1, 1),
        tukey_alpha,
    )
    im_center = np.asarray(image_data.shape) / 2
    trans1, trans2 = np.round(centers - im_center)
    output1 = Image(
        data=im1,
        scale=image.scale,
        translate=image.data_to_world(trans1),
        name=image.name + " 1",
    )
    output2 = Image(
        data=im2,
        scale=image.scale,
        translate=image.data_to_world(trans2),
        name=image.name + " 2",
    )

    return [output1, output2]
