"""Some stuff are adapted from:
https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py.
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import center_of_mass
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_otsu
from tqdm import tqdm

from ..utils import center_to_corners, load_array, summary

# +------------------------------------------------------------------------------------------+ #
#                                           Volume                                           | #
# +------------------------------------------------------------------------------------------+ #


def compute_volume(boxes: np.ndarray) -> np.ndarray:
    """Compute the volume of each boxes in a given list.

    Args:
        boxes (np.ndarray): List of boxes. Array of shape (num_boxes, (x1, y1, z1, x2, y2, z2)).

    Returns:
        np.ndarray: (num_boxes, ): Volume of each boxe.
    """
    x1, y1, z1 = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    x2, y2, z2 = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    return (x2 - x1) * (y2 - y1) * (z2 - z1)


# +------------------------------------------------------------------------------------------+ #
#                                             IoU                                            | #
# +------------------------------------------------------------------------------------------+ #


def compute_iou_3d(
    box: np.ndarray, boxes: np.ndarray, box_volume: np.ndarray, boxes_volume: np.ndarray
) -> np.ndarray:
    """Calculates IoU of the given box with the array of the given boxes.

    Args:
        box (np.ndarray): 1D vector (x1, x1, z1, x2, y2, z2).
        boxes (np.ndarray): List of boxes. Array of shape (num_boxes, (x1, y1, z1, x2, y2, z2)).
        box_volume (np.ndarray): Volume of the given box: (x2 - x1) * (y2 - y1) * (z2 - z1)
        boxes_volume (np.ndarray): Volume of all the given boxes. 1D vector (num_boxes, ).

    Returns:
        np.ndarray: IoU between box and every other box in boxes. 1D vector (num_boxes, ).
                    Each element is a float32 in [0, 1].
    """
    x1, x2 = np.maximum(box[0], boxes[:, 0]), np.minimum(box[3], boxes[:, 3])
    y1, y2 = np.maximum(box[1], boxes[:, 1]), np.minimum(box[4], boxes[:, 4])
    z1, z2 = np.maximum(box[2], boxes[:, 2]), np.minimum(box[5], boxes[:, 5])
    intersection = (
        np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    )
    union = box_volume + boxes_volume[:] - intersection[:]
    iou = intersection / union
    return iou


# +------------------------------------------------------------------------------------------+ #
#                                         Group Predictions                                  | #
# +------------------------------------------------------------------------------------------+ #


def group_predictions(boxes: np.ndarray) -> Tuple[List[np.ndarray]]:
    """Group predictions (boxes (+ scores)) that have a non zero IoU. This is a kind of
    connected components retrieval on boxes.
    Note that this is almost NMS itself with a threshold of 0 except that we don't take
    scores into account and remember the found group at each iteration.

    Args:
        boxes (np.ndarray): List of boxes. Array of shape (num_boxes, (x1, y1, z1, x2, y2, z2)).

    Returns:
        Tuple[List[np.ndarray]]: Grouped boxes and predictions. Each boxes group and scores group
                                 are of shape (k, (x1, y1, z1, x2, y2, z2)) and (k, ).
                                 The length of all groups sum up to num_boxes.
    """
    volumes = compute_volume(boxes)
    boxes_groups = []
    while len(boxes) > 0:
        reference_bbox = boxes[0]
        reference_volume = volumes[0]
        ious = compute_iou_3d(reference_bbox, boxes, reference_volume, volumes)
        group_indices = np.where(ious > 0)[0]
        boxes_groups.append(boxes[group_indices])
        boxes = np.delete(boxes, group_indices, axis=0)
        volumes = np.delete(volumes, group_indices)
    return boxes_groups


# +------------------------------------------------------------------------------------------+ #
#                                   Spatial Average Correction                               | #
# +------------------------------------------------------------------------------------------+ #


def spatial_average(grouped_boxes: List[np.ndarray]) -> np.ndarray:
    """Compute the mean boxes of each group in grouped_boxes.

    Args:
        grouped_boxes(List[np.ndarray]): List of group of boxes: (N, k, (x1, y1, z1, x2, y2, z2)).
                                         N is the number of groups, k_i is the number of boxes in
                                         the i-th group.

    Returns:
        np.ndarray: Mean boxe of each group: (N, (x1, y1, z1, x2, y2, z2)).
    """
    mean_boxes = np.ndarray((len(grouped_boxes), 6))
    for i, group in enumerate(grouped_boxes):
        x1, y1, z1 = group[:, 0], group[:, 1], group[:, 2]
        x2, y2, z2 = group[:, 3], group[:, 4], group[:, 5]
        mean_boxes[i] = np.array(
            (x1.mean(), y1.mean(), z1.mean(), x2.mean(), y2.mean(), z2.mean())
        )
    return mean_boxes


# +------------------------------------------------------------------------------------------+ #
#                                    Non Maximum Suppresion                                  | #
# +------------------------------------------------------------------------------------------+ #


def non_max_suppression_3d(
    boxes: np.ndarray, scores: np.ndarray, threshold: float
) -> np.ndarray:
    """Performs 3D non-maximum suppression and returns indices of kept boxes.

    Args:
        boxes (np.ndarray): List of boxes. Array of shape (num_boxes, (x1, y1, z1, x2, y2, z2)).
        scores (np.ndarray): 1-D array of box scores. (num_boxes, )
        threshold (float): IoU threshold to use for filtering. Float in [0, 1].

    Returns:
        np.ndarray: Indices of kept boxes. (k, ) where 0 < k < num_boxes.
    """
    # Compute box volumes
    volume = compute_volume(boxes)
    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou_3d(boxes[i], boxes[ixs[1:]], volume[i], volume[ixs[1:]])
        # Identify boxes with IoU over the threshold. This returns indices into ixs[1:],
        # so add 1 to get indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


# +------------------------------------------------------------------------------------------+ #
#                                   Center of mass correction                                | #
# +------------------------------------------------------------------------------------------+ #


def get_center_of_mass(image: np.ndarray) -> np.ndarray:
    """Compute the center of mass of an image.
    Note that the center of mass is computed on a binary image obtained by Otsu thresholding.

    Args:
        image (np.ndarray): The 3D image array: (depth, height, width).

    Returns:
        np.ndarray: Coordinates of the center of mass: (center_z, center_y, center_x).
    """
    binary_image = image > threshold_otsu(image)
    return center_of_mass(binary_image)


def center_of_mass_correction(bbox: np.ndarray, image: np.ndarray) -> np.ndarray:
    """For one given bbox, compute the center of mass of its binary mask obtained by otsu
    thresholding, and create a new box centered on it.

    Args:
        bbox (np.ndarray): Array (x_min, y_min, z_min, x_max, y_max, z_max)
        image (np.ndarray): 3D Image into which boxes indicates objects: (depth, height, width).

    Returns:
        np.ndarray: New recentered bbox (x_min, y_min, z_min, x_max, y_max, z_max).
    """
    x_min, y_min, z_min, x_max, y_max, z_max = bbox.round().astype(int)
    crop = image[max(0, z_min) : z_max, max(0, y_min) : y_max, max(0, x_min) : x_max]
    center_relative = get_center_of_mass(crop)  # coordinates inside the crop
    center_absolute = np.array(center_relative) + np.array((z_min, y_min, x_min))
    depth, height, width = z_max - z_min, y_max - y_min, x_max - x_min
    corners = center_to_corners(center_absolute, (depth, height, width))
    return np.array(corners)


def iterative_center_of_mass_correction(
    bbox: np.ndarray,
    image: np.ndarray,
    min_distance: float = 0.1,
    n_iter_max: int = 1000,
) -> np.ndarray:
    d, h, w = image.shape
    # State initialization: two elements: center and distance
    x_min, y_min, z_min, x_max, y_max, z_max = bbox
    depth, height, width = z_max - z_min, y_max - y_min, x_max - x_min
    old_center = np.array((z_min + depth // 2, y_min + height // 2, x_min + width // 2))
    distance = min_distance + 1  # init so that first while test is True
    n_iter = 0
    while distance > min_distance and n_iter < n_iter_max:
        # compute new bouding box
        bbox = center_of_mass_correction(bbox, image)
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        depth, height, width = z_max - z_min, y_max - y_min, x_max - x_min
        # stop and revert previous step if new center is too close to the border
        if x_min < 0 or y_min < 0 or z_min < 0 or x_max > w or y_max > h or z_max > d:
            bbox = center_to_corners(old_center, (depth, height, width))
            break
        # update state
        new_center = np.array(
            (z_min + depth // 2, y_min + height // 2, x_min + width // 2)
        )
        distance = np.linalg.norm(new_center - old_center)
        old_center = new_center
        n_iter += 1
    return bbox


def center_of_mass_correction_on_all_bboxes(
    boxes: np.ndarray,
    image: np.ndarray,
    iterative: bool = True,
    min_distance: float = 0.1,
) -> np.ndarray:
    """For each given boxes, compute its center of mass (iteratively or not) and create a new
    box centered on it.

    Args:
        boxes (np.ndarray): List of boxes. Array of shape (num_boxes, (x1, y1, z1, x2, y2, z2)).
        image (np.ndarray): 3D Image into which boxes indicates objects: (depth, height, width).
        iterative (bool, optional): If True, the center of mass correction is applied iteratively
                                    until a stopping criterion is met. Default to True.
        min_distance (float, optional): If the center of mass correction is applied iteratively,
                                        it is so until the new center is at most at min_distance
                                        from the old one.

    Returns:
        np.ndarray: Corners array, of shape (num_boxes, (x1, y1, z1, x2, y2, z2)).
    """
    correction = (
        iterative_center_of_mass_correction if iterative else center_of_mass_correction
    )
    correction_kwargs = {"min_distance": min_distance} if iterative else {}
    corrected_boxes = []
    for bbox in boxes:
        corrected_bbox = correction(bbox, image, **correction_kwargs)
        if not np.isnan(corrected_bbox).any():
            corrected_boxes.append(corrected_bbox)
    return np.array(corrected_boxes)


def otsu_threshold(image: np.ndarray, boxes: np.ndarray, margin: float) -> np.ndarray:
    """Filtered boxes by computing a threshold via Otsu method on boxes intensities, after
    having corrected the image histogramm via CLAHE method.

    Args:
        image (np.ndarray): The image on wich CLAHE will be applied to compute boxes intensities.
        boxes (np.ndarray): List of boxes. Array of shape (num_boxes, (x1, y1, z1, x2, y2, z2)).
        margin (float): A box will be kept if its intensity is > margin * threshold.

    Returns:
        np.ndarray: Filtered corners array, of shape (new_num_boxes, (x1, y1, z1, x2, y2, z2)),
                    Where new_num_boxes <= num_boxes.
    """
    # 1. Correct image histogramm
    corrected_image = equalize_adapthist(image)
    # 2. Fetch boxes intensities
    intensities = []
    for corners in boxes:
        x_min, y_min, z_min, x_max, y_max, z_max = corners.astype(int)
        patch = corrected_image[z_min:z_max, y_min:y_max, x_min:x_max]
        intensities.append(patch.sum())
    intensities = np.array(intensities)
    # 3. Normalize intensities
    intensities = (intensities - intensities.min()) / (
        intensities.max() - intensities.min()
    )
    # 4. Compute Otsu threshold
    threshold = threshold_otsu(intensities)
    # 5. Filter boxes
    filtered_boxes = boxes[intensities > margin * threshold]
    return filtered_boxes


# +------------------------------------------------------------------------------------------+ #
#                            Refinements & Postprocessing Wrappers                           | #
# +------------------------------------------------------------------------------------------+ #


def refine(
    image: np.ndarray,
    boxes: np.ndarray,
    iterative: bool = True,
    min_distance: float = 0.1,
    last_step: str = "spatial_average",
    margin: float = 0.5,
) -> Tuple[np.ndarray]:
    """Being given an image and its predicted boxes, compute a set of refined boxes, that is:
        1. spatial average of each connected components
        2. iterative center of mass correction
        3.  * spatial average of each connected components
            OR
            * Otsu thresholding of boxes intensities computed on a image corrected by
           Contrast Limited Adaptive Histogram Equalization (CLAHE)
    Args:
        image (np.ndarray): 3D image being predicted: (depth, height, width).
        boxes (np.ndarray): Predicted bounding boxes: (num_boxes, (x1, y1, z1, x2, y2, z2)).
        iterative (bool, optional): If True, the center of mass correction is applied iteratively
                                    until a stopping criterion is met. Default to True.
        min_distance (float, optional): If the center of mass correction is applied iteratively,
                                        it is so until the new center is at most at min_distance
                                        from the old one.
        last_step (str): One of 'spatial_average' or 'otsu'. Determines which last postprocessing
                         step to applied after spatial average and center of mass correction.

    Returns:
        Tuple[np.ndarray]: Three corners array, each of shape (n, (x1, y1, z1, x2, y2, z2)),
                           after each postprocessing step.
    """
    # 1. Spatial Average of Connected components
    step1 = spatial_average(group_predictions(boxes))
    # 2. (Iterative) Center of mass Correction
    correction_args = [step1, image, iterative, min_distance]
    step2 = center_of_mass_correction_on_all_bboxes(*correction_args)
    # 3. Spatial Average of Connected components on corrected boxes OR Ostu after CLAHE
    if last_step == "otsu":
        step3 = otsu_threshold(image, step2, margin)
    else:
        step3 = spatial_average(group_predictions(step2))
    return step1, step2, step3


# +------------------------------------------------------------------------------------------+ #
# |                                MAIN POSTPROCESSING FUNCTION                              | #
# +------------------------------------------------------------------------------------------+ #


def postprocess_one_image(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    nms_threshold: float = 0.1,
    iterative: bool = True,
    last_step: str = "spatial_average",
    margin: float = 0.5,
) -> Dict:
    """Being given an image, bounding boxes, and scores, return a set of 'good' bounding boxes.
    The bounding boxes are filtered based on the following procedure:
        1. Non Maximum Suppresion
        2. Spatial Average of each Connected Components
        3. (Iterative) Center of mass correction, where the center of mass is computer on
           a binary mask obtained by otsu thresholding.
        4. Spatial Average of each resulting Connected Components

    Args:
        image (np.ndarray): 3D image being predicted: (depth, height, width). The image must
                              be padded the same way it was during inference, with respect to
                              given patch size and stride.
        boxes (np.ndarray): Predicted bounding boxes: (num_boxes, (x1, y1, z1, x2, y2, z2)).
        scores (np.ndarray): Confidence score of each bouding boxes: (num_boxes, ).
        nms_threshold (float, optional): IoU threshold above which it is considered that
                                         predictions overlap. Used in the NMS algorithm.
                                         Defaults to 0.1.
        iterative (bool, optional): If True, the center of mass correction is applied iteratively
                                    until a stopping criterion is met. Default to True.
        min_distance (float, optional): If the center of mass correction is applied iteratively,
                                        it is so until the new center is at most at min_distance
                                        from the old one.

    Returns:
        Dict: Corners array, of shape (num_boxes', (x1, y1, z1, x2, y2, z2)), with
              num_boxes' < num_boxes, after each postprocessing steps.
    """
    # NMS
    filtered_indices = non_max_suppression_3d(boxes, scores, threshold=nms_threshold)
    filtered_boxes = boxes[filtered_indices]
    # Refinement
    step1, step2, step3 = refine(
        image, filtered_boxes, iterative, nms_threshold, last_step, margin
    )
    results = {
        "raw": boxes,
        "nms": filtered_boxes,
        "first_spatial_average": step1,
        "center_of_mass_correction": step2,
        "last_step": step3,
    }
    return results


# +------------------------------------------------------------------------------------------+ #
# |                                       EXTERNAL CALL                                      | #
# +------------------------------------------------------------------------------------------+ #


def postprocess(
    predictions_path: str,
    rootdir: str,
    patch_size: Tuple[int],
    stride: Tuple[int],
    nms_threshold: float,
    iterative: bool,
    last_step: str,
    margin: float,
    extension: str,
    testset_size: int = None,
    image_name: str = None,
    output_dir: str = None,
    downscale: float = 1.0,
) -> Dict[str, Tuple[np.ndarray]]:
    """If image_name is None, postprocess whole folder."""
    patch_size = tuple(np.rint(np.array(patch_size) / downscale).astype(int))
    stride = (
        np.array(patch_size) // 4
        if stride is None
        else tuple(np.rint(np.array(stride) / downscale).astype(int))
    )
    summary_output = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        summary_output = os.path.join(output_dir, "postprocessing_config.txt")
    summary(locals(), title="POSTPROCESSING", output=summary_output)
    with open(predictions_path, "rb") as file:
        predictions = pickle.load(file)
    if image_name is not None:
        images_names = [image_name]
    else:
        images_names = sorted(
            list(filter(lambda x: x.endswith(extension), os.listdir(rootdir)))
        )
        images_names = images_names[:testset_size]
    postprocess_kwargs = {
        "nms_threshold": nms_threshold,
        "iterative": iterative,
        "last_step": last_step,
        "margin": margin,
    }
    results = {}
    for image_name in tqdm(images_names):
        centers, scores = predictions[image_name]
        corners = np.array(
            [center_to_corners(center, patch_size) for center in centers]
        )
        image = load_array(os.path.join(rootdir, image_name))
        results[image_name] = postprocess_one_image(
            image, corners, scores, **postprocess_kwargs
        )
    if output_dir is not None:
        with open(os.path.join(output_dir, "raw_predictions.pickle"), "wb") as file:
            pickle.dump(predictions, file)
        with open(os.path.join(output_dir, "predictions.pickle"), "wb") as file:
            pickle.dump(results, file)
    return results
