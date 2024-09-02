import os
import pickle
from typing import Dict, List, Tuple

import imageio
import numpy as np
from skimage import img_as_ubyte
from sklearn.metrics import classification_report, confusion_matrix

from ..utils import center_to_corners, load_annotations, load_array, summary
from .postprocess import compute_iou_3d, compute_volume

# +------------------------------------------------------------------------------------------+ #
# |                                       LABEL VOLUMES                                      | #
# +------------------------------------------------------------------------------------------+ #


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    return np.stack((image,) * 3, axis=-1)


def recenter_if_needed(shape: Tuple[int], corners: np.ndarray) -> Tuple[int]:
    D, H, W = shape
    x_min, y_min, z_min, x_max, y_max, z_max = corners.astype(np.int)
    d, h, w = z_max - z_min, y_max - y_min, x_max - x_min
    z_min, y_min, x_min = max(0, z_min), max(0, y_min), max(0, x_min)
    z_max, y_max, x_max = z_min + d, y_min + h, x_min + w
    z_max, y_max, x_max = min(D, z_max), min(H, y_max), min(W, x_max)
    z_min, y_min, x_min = z_max - d, y_max - h, x_max - w
    return x_min, y_min, z_min, x_max, y_max, z_max


def draw_bbox(
    image: np.ndarray, bbox: np.ndarray, color: Tuple[int] = (255, 0, 0)
) -> None:
    """Modify image inplace.
    This function draws a 3D bbox to be seen from the z-axis, that is a 2D bbox
    [x_min, x_max, y_min, y_max] at each z in range [z_min, z_max].
    """
    x_min, y_min, z_min, x_max, y_max, z_max = recenter_if_needed(image.shape[:3], bbox)
    image[z_min:z_max, y_min, x_min:x_max] = color
    image[z_min:z_max, y_max, x_min:x_max] = color
    image[z_min:z_max, y_min:y_max, x_min] = color
    image[z_min:z_max, y_min:y_max, x_max] = color


def draw_bboxes(
    image: np.ndarray, bboxes: np.ndarray, color: Tuple[int] = (255, 0, 0)
) -> None:
    """Modify image inplace."""
    for bbox in bboxes:
        draw_bbox(image, bbox, color)


def save_3d_image_to_tiff(image: np.ndarray, name: str) -> None:
    imageio.mimwrite(f"{name}.tiff", image)


def create_and_save_labelled_volume(
    predicted_corners: np.ndarray,
    true_corners: np.ndarray,
    tp: List[int],
    fp: List[int],
    fn: List[int],
    image_dir: str,
    image_name: str,
    output_dir: str,
) -> None:
    # 1. Load and prepare image array
    image = load_array(os.path.join(image_dir, image_name))
    image = grayscale_to_rgb(img_as_ubyte(image))
    # 2. Draw colored groundtruth and predictions
    colors = {"red": (255, 0, 0), "green": (0, 255, 0), "yellow": (255, 255, 0)}
    draw_bboxes(image, true_corners[fn], color=colors["red"])
    draw_bboxes(image, predicted_corners[tp], color=colors["green"])
    draw_bboxes(image, predicted_corners[fp], color=colors["yellow"])
    # 3. Save results to npz and tiff
    output_prefix = os.path.join(output_dir, image_name)
    np.savez(output_prefix, image=image)
    save_3d_image_to_tiff(image, output_prefix)


# +------------------------------------------------------------------------------------------+ #
# |                                    METRICS COMPUTATION                                   | #
# +------------------------------------------------------------------------------------------+ #


def get_predictions_indices(
    predictions: np.ndarray,
    groundtruth: np.ndarray,
    threshold: float,
) -> Dict:
    true_positive, false_positive, false_negative = [], [], []
    predictions_volume = compute_volume(predictions)
    groundtruth_volume = compute_volume(groundtruth)
    for i, prediction in enumerate(predictions):
        ious = compute_iou_3d(
            prediction, groundtruth, predictions_volume[i], groundtruth_volume
        )
        if ious.max() < threshold:
            false_positive.append(i)
    for i, gt in enumerate(groundtruth):
        ious = compute_iou_3d(
            gt, predictions, groundtruth_volume[i], predictions_volume
        )
        if ious.max() > threshold:
            true_positive.append(ious.argmax())
        else:
            false_negative.append(i)
    return true_positive, false_positive, false_negative


def evaluate_one_image_picking(
    predicted_corners: np.ndarray,
    true_corners: np.ndarray,
    threshold: float,
    label_volume: bool,
    image_dir: str = None,
    image_name: str = None,
    output_dir: str = None,
) -> Dict:
    tp, fp, fn = get_predictions_indices(predicted_corners, true_corners, threshold)
    if label_volume:
        label_args = [predicted_corners, true_corners, tp, fp, fn]
        io_args = [image_dir, image_name, output_dir]
        create_and_save_labelled_volume(*label_args, *io_args)
    true_positive, false_positive, false_negative = len(tp), len(fp), len(fn)
    precision = true_positive / (true_positive + false_positive + 1e-20)
    recall = true_positive / (true_positive + false_negative + 1e-20)
    f1 = 2 * precision * recall / (precision + recall + 1e-20)
    metrics = {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics


# +------------------------------------------------------------------------------------------+ #
# |                                      EXTERNAL CALLS                                      | #
# +------------------------------------------------------------------------------------------+ #


def evaluate_picking(
    predictions_path: str,
    annotations_path: str,
    patch_size: str,
    threshold: float,
    step: str,
    make_labelled_volumes: bool,
    image_dir: str = None,
    image_name: str = None,
    output_dir: str = None,
) -> Dict[str, Dict[str, float]]:
    """If image_name is None, evaluate all predicted images.
    predictions[image_name] = {
        'raw':                       (N1', (x1, y1, z1, x2, y2, z2)),
        'nms':                       (N2', (x1, y1, z1, x2, y2, z2)),
        'first_spatial_average':     (N3', (x1, y1, z1, x2, y2, z2)),
        'center_of_mass_correction': (N4', (x1, y1, z1, x2, y2, z2)),
        'last_spatial_average':      (N5', (x1, y1, z1, x2, y2, z2)),
    }
    with N1 > N2 > N3 = N4 > N5.
    See postprocess.py.
    """
    summary_output = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        summary_output = os.path.join(output_dir, "evaluation_config.txt")
    summary(locals(), title="EVALUATION", output=summary_output)
    with open(predictions_path, "rb") as file:
        predictions = pickle.load(file)
    annotations = load_annotations(annotations_path)
    images_names = [image_name] if image_name is not None else predictions.keys()
    evaluations = {}
    volumes_output_dir = None
    if make_labelled_volumes:
        volumes_output_dir = os.path.join(output_dir, "labelled_volumes")
        os.makedirs(volumes_output_dir, exist_ok=True)
    for image_name in images_names:
        true_centers = annotations[annotations[:, 0] == image_name, 1:]
        true_corners = [
            center_to_corners(center, patch_size) for center in true_centers
        ]
        true_corners = np.array(true_corners)
        predicted_corners = predictions[image_name][step]
        metrics_args = [predicted_corners, true_corners, threshold]
        label_args = [make_labelled_volumes, image_dir, image_name, volumes_output_dir]
        evaluations[image_name] = evaluate_one_image_picking(*metrics_args, *label_args)
    mean_metrics_over_testset = {}
    for metric in [
        "true_positive",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1",
    ]:
        values = [evaluations[image_name][metric] for image_name in images_names]
        mean_metrics_over_testset[metric] = np.array(values).mean()
    if output_dir is not None:
        summary_output = os.path.join(output_dir, "evaluation_results.txt")
        summary(
            mean_metrics_over_testset,
            title="EVALUATION MEAN RESULTS",
            output=summary_output,
        )
        with open(os.path.join(output_dir, "evaluations.pickle"), "wb") as file:
            pickle.dump(evaluations, file)
    return evaluations


def flatten_dict(dd, separator=".", prefix=""):
    # https://stackoverflow.com/a/19647596
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


def evaluate_tilt(
    predictions_path: str,
    annotations_path: str,
    image_name: str = None,
    output_dir: str = None,
    selected_particules: str = None,  # particules from the training set
) -> Dict[str, Dict[str, float]]:
    if selected_particules is not None:
        with open(selected_particules, "rb") as f:
            train_particles = pickle.load(f)
    else:
        train_particles = None
    summary_output = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        summary_output = os.path.join(output_dir, "evaluation_config.txt")
    # summary(locals(), title='EVALUATION', output=summary_output)
    with open(predictions_path, "rb") as file:
        predictions = pickle.load(file)
    annotations = load_annotations(annotations_path)
    images_names = [image_name] if image_name is not None else predictions.keys()
    evaluations = {}
    for image_name in images_names:
        if train_particles is not None:
            train_particles_image = train_particles[
                train_particles[:, 1] == image_name, 0
            ]
        else:
            train_particles_image = []
        true_views = (
            annotations[annotations[:, 0] == image_name, 1:].reshape(-1).astype(int)
        )
        predicted_views = predictions[image_name][0].astype(int)
        if predicted_views.shape[0] > true_views.shape[0]:
            predicted_views = predicted_views[: true_views.shape[0]]
        evaluations[image_name] = {}
        evaluations[image_name]["train"] = np.array(
            [i in train_particles_image for i in range(len(true_views))], dtype=bool
        )  # train mask
        evaluations[image_name]["true"] = true_views
        evaluations[image_name]["pred"] = predicted_views

    eval = {
        k: np.concatenate([evaluations[im][k] for im in images_names])
        for k in ["train", "true", "pred"]
    }

    y_test_true = eval["true"][~eval["train"]]
    y_train_true = eval["true"][eval["train"]]
    y_test_pred = eval["pred"][~eval["train"]]
    y_train_pred = eval["pred"][eval["train"]]

    if selected_particules is not None:
        cr_train = classification_report(
            y_train_true,
            y_train_pred,
            labels=[0, 1, 2],
            target_names=["top", "side", "other"],
        )
        cm_train = confusion_matrix(y_train_true, y_train_pred, labels=[0, 1, 2])
    cr_test = classification_report(
        y_test_true,
        y_test_pred,
        labels=[0, 1, 2],
        target_names=["top", "side", "other"],
    )
    cm_test = confusion_matrix(y_test_true, y_test_pred, labels=[0, 1, 2])

    if output_dir is not None:
        summary_output = os.path.join(output_dir, "evaluation_results.txt")
        with open(summary_output, "w") as f:
            if selected_particules is not None:
                f.write("TRAIN SET\n\n")
                f.write(str(cm_train) + "\n\n")
                f.write(cr_train + "\n\n")
            f.write("TEST SET\n\n")
            f.write(str(cm_test) + "\n\n")
            f.write(cr_test + "\n\n")

        with open(os.path.join(output_dir, "evaluations.pickle"), "wb") as file:
            pickle.dump(evaluations, file)
    return evaluations
