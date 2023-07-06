import os

rootdir = "./examples/data/Nicollo_STED"
output_picking = "./examples/output_picking"
images_dir = "./examples/data/Nicollo_STED/raw"
ext = "tif"
patch_size = "15 24 24"
dim = 3

# Train
epochs = 5
epoch_size = 20
batch_size = 128
radius = 15
n_particles_per_image = 35

# Predict
stride = 10
"""
if os.path.exists(output_picking):
    os.system(f'rm -r {output_picking}')

base_command = f"python -m spfluo.picking --rootdir {rootdir} --output_dir {output_picking} --patch_size {patch_size} --ext {ext} --dim {dim}"
os.system(base_command+" --stages prepare --crop_output_dir cropped")
#os.system(base_command+f" --stages train --mode fs --num_epochs {epochs} --batch_size 8 --num_workers 4 --augment 0.8")
os.system(base_command+f" --stages train --shuffle --mode pu --epoch_size {epoch_size} --radius {radius} --num_particles_per_image {n_particles_per_image} --num_epochs {epochs} --batch_size {batch_size} --num_workers 8 --augment 0.8")
os.system(base_command+f" --stages predict --testdir {images_dir} --checkpoint {os.path.join(output_picking,'checkpoint.pt')} --stride {stride}")
os.system(base_command+f" --stages postprocess --testdir {images_dir} --predictions {os.path.join(output_picking, 'predictions.pickle')} --stride {stride}")
"""
# Crop out picking results
import pickle
import numpy as np
import imageio
import tifffile
from skimage import io

patch_size = "32 32 32"

with open(os.path.join(output_picking, "predictions.pickle"), "rb") as f:
    predictions = pickle.load(f)
for image_name in predictions.keys():
    image_path = os.path.join(images_dir, image_name)
    image = np.stack(imageio.mimread(image_path, memtest=False)).astype(np.int16)
    padding = tuple(map(int, patch_size.split(" ")))
    padding = tuple([(p // 2, p // 2) for p in padding])
    image_padded = np.pad(image, padding)
    image_padded_annotated = np.stack((image_padded,) * 3, axis=0)
    color = np.array((0, 255, 0))[:, None, None]
    D, H, W = image.shape
    dir_path = os.path.join(output_picking, "ee" + image_name)
    os.mkdir(dir_path)
    for i, bbox in enumerate(predictions[image_name]["last_step"]):
        bbox = np.rint(bbox).astype(int)
        x1, y1, z1, x2, y2, z2 = bbox
        x1 += padding[2][0]
        y1 += padding[1][0]
        z1 += padding[0][0]
        x2 += padding[2][0]
        y2 += padding[1][0]
        z2 += padding[0][0]
        patch = image_padded[z1:z2, y1:y2, x1:x2]
        patch = patch.astype(float)
        if 0 not in patch.shape:
            patch = (patch - patch.min()) / (patch.max() - patch.min())
            patch = (patch * 255).astype(np.uint8)
            tifffile.imwrite(os.path.join(dir_path, f"patch_{i}.{ext}"), patch)
            image_padded_annotated[:, z1:z2, y1, x1:x2] = color
            image_padded_annotated[:, z1:z2, y2, x1:x2] = color
            image_padded_annotated[:, z1:z2, y1:y2, x1] = color
            image_padded_annotated[:, z1:z2, y1:y2, x2] = color

    tifffile.imwrite(
        os.path.join(dir_path, f"annotated.{ext}"),
        np.swapaxes(image_padded_annotated, 0, 1).astype(np.int8),
        photometric="rgb",
        imagej=True,
    )
    # io.imsave(os.path.join(dir_path, f'annotated.{ext}'), image_padded_annotated)
