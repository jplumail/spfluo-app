import json
import os
import shutil
import sys
from glob import glob
from typing import Optional, Tuple

import numpy as np
import tifffile
from scipy.ndimage import zoom

from ..utils import load_annotations, load_array


def read_center_and_size(file_path):
    with open(file_path) as f:
        data = json.load(f)
    center = data["markups"][0]["center"]
    size = data["markups"][0]["size"]
    return center, size


def read_rois(list_paths):
    centers, sizes = [], []
    for fp in list_paths:
        center, size = read_center_and_size(fp)
        centers.append(center)
        sizes.append(size)
    roi_sizes = np.amax(np.ceil(np.stack(sizes)), axis=1)
    return centers, roi_sizes


def create_annotation_image(
    rootdir: str,
    annotations_path: str,
    patch_size: Tuple[int],
    color: Tuple[int],
    extension: str,
) -> None:
    """
    Annotate images from a root directory with centers of bounding boxes stored in a CSV file.
    Arguments:
        - rootdir (str) : the root directory where are stored the images
        - annotations_path (str) : the path to the CSV file storing the centers
                                   in the format "image_name.tif,i,z,y,x"
        - patch_size (Tuple[int]) : the edge length of the bounding box (will be rounded to an even length if odd)
        - color (Tuple[int]) : RGB color of the bbox
        - extension (str) : format of the image files. Ex: tiff, npz, tif
    """
    image_list = list(filter(lambda x: x.endswith(extension), os.listdir(rootdir)))
    annotations = load_annotations(annotations_path)
    for image_name in image_list:
        im_path = os.path.join(rootdir, image_name)
        im = load_array(im_path)
        if im.ndim == 3:
            im = np.stack([im, im, im], axis=-1)
        color = np.array(color).astype(im.dtype)

        # Draw a cube for each ROI
        centers = annotations[annotations[:, 0] == os.path.basename(im_path), 2:]
        for center in centers:
            z, y, x = np.rint(np.array(center, dtype=float)).astype(int)
            x1, x2 = x - patch_size[2] // 2, x + patch_size[2] // 2
            y1, y2 = y - patch_size[1] // 2, y + patch_size[1] // 2
            z1, z2 = z - patch_size[0] // 2, z + patch_size[0] // 2
            im[z1:z2, y1, x1:x2] = color
            im[z1:z2, y2, x1:x2] = color
            im[z1:z2, y1:y2, x1] = color
            im[z1:z2, y1:y2, x2] = color

        tifffile.imwrite(im_path, im)


def extract_annotations(
    slicer3d_dir: str,
    rootdir: str,
    outputdir: str,
    patch_size: Optional[int] = None,
    downscale: float = 1.0,
):
    os.makedirs(rootdir, exist_ok=True)
    coordinates = []
    list_centers = []
    list_roi_sizes = []
    list_img_paths = []
    for d in os.listdir(slicer3d_dir):
        im_dir = os.path.join(slicer3d_dir, d)

        # Copy image
        im_path = glob(os.path.join(im_dir, "*.tif")) + glob(
            os.path.join(im_dir, "*.tiff")
        )
        if len(im_path) > 1:
            print("Too many TIF in a directory", file=sys.stderr)
        if len(im_path) > 0:
            print(
                f"Current directory: {os.path.join(slicer3d_dir, d)}", file=sys.stdout
            )
            im_path = im_path[0]
            im_name = os.path.basename(im_path)
            im_outputpath = os.path.join(rootdir, im_name)
            print(f"Found image {im_path}, copying to {im_outputpath}", file=sys.stdout)
            shutil.copyfile(im_path, im_outputpath)
            if downscale > 1:
                print(f"Downscaling with factor {downscale}...", file=sys.stdout)
                im = load_array(im_outputpath)
                im = zoom(im, 1 / downscale)
                tifffile.imwrite(im_outputpath, im)

            # Process annotations
            ann_paths = glob(os.path.join(im_dir, "*.json"))
            if len(ann_paths) > 0:
                print(f"Found {len(ann_paths)}, processing...", file=sys.stdout)
                centers, roi_sizes = read_rois(ann_paths)
                if downscale > 1:
                    centers = [[c / downscale for c in center] for center in centers]
                    roi_sizes = [s / downscale for s in roi_sizes]
                list_img_paths.append(im_path)
                list_centers.append(centers)
                list_roi_sizes.append(roi_sizes)
                for i, center in enumerate(centers):
                    coordinates.append(
                        [
                            im_name,
                            str(i),
                            str(float(center[2])),
                            str(float(center[1])),
                            str(float(center[0])),
                        ]
                    )

    # Infer patch size if not provided
    if patch_size is None:
        ps = int(np.max(list_roi_sizes))
        print(f"No patch size provided, chosing {ps}.", file=sys.stdout)
    else:
        ps = patch_size
        if isinstance(ps, list):
            ps = max(ps)

    # Save annotations
    print(
        f"Saving annotations at {os.path.join(rootdir, 'coordinates.csv')}",
        file=sys.stdout,
    )
    with open(os.path.join(rootdir, "coordinates.csv"), "w") as f:
        for coord in coordinates:
            f.write(",".join(coord))
            f.write("\n")
