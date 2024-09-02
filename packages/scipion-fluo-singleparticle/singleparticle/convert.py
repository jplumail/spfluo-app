from __future__ import annotations

import csv
import glob
import itertools
import math
import os
from typing import Dict, Iterator, List, Tuple

import numpy as np
from pwfluo.objects import (
    Coordinate3D,
    Image,
    Particle,
    SetOfCoordinates3D,
    SetOfParticles,
    Transform,
)
from scipy.spatial.transform import Rotation
from spfluo.utils.volume import interpolate_to_size, resample


def getLastParticlesParams(folder):
    # Read poses
    fpaths = glob.glob(os.path.join(folder, "estimated_poses_epoch_*.csv"))
    poses_path = sorted(fpaths, key=lambda x: int(x.split("_")[-1][:-4]), reverse=True)[
        0
    ]
    output: dict[str, dict[str, np.ndarray]] = {}
    with open(poses_path, "r") as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            rot = np.array(list(map(float, row[1:4])))
            trans = np.array(list(map(float, row[4:7])))
            H = np.zeros((4, 4), dtype=float)
            H[:3, :3] = Rotation.from_euler("XZX", rot, degrees=True).as_matrix()
            H[:3, 3] = trans
            output[row[0]] = {"homogeneous_transform": H}
    return output


def updateSetOfParticles(
    inputSetOfParticles: SetOfParticles,
    outputSetOfParticles: SetOfParticles,
    particlesParams: Dict,
) -> None:
    """Update a set of particles from a template
    and copy attributes coverage/score/transform"""

    def updateParticle(particle: Particle, index: int):
        particleParams = particlesParams.get(
            particle.getBaseName().replace(".ome.tiff", ".tiff")
        )
        if not particleParams:
            setattr(particle, "_appendItem", False)
        else:
            # Create 4x4 matrix from 4x3 e2spt_sgd align matrix and append row [0,0,0,1]
            H = np.array(particleParams["homogeneous_transform"])
            particle.setTransform(Transform(H))

    outputSetOfParticles.copyItems(
        inputSetOfParticles,
        updateItemCallback=updateParticle,
        itemDataIterator=itertools.count(0),
    )
    outputSetOfParticles.write()


def readSetOfParticles(
    imageFile: str,
    particleFileList: List[str],
    outputParticlesSet: SetOfParticles,
    coordSet: List[Coordinate3D],
) -> SetOfParticles:
    for counter, particleFile in enumerate(particleFileList):
        particle = Particle(data=particleFile)
        particle.setCoordinate3D(coordSet[counter])
        coord = coordSet[counter]
        transformation = coord._transform
        shift_x, shift_y, shift_z = transformation.getShifts()
        transformation.setShifts(shift_x, shift_y, shift_z)
        particle.setTransform(transformation)
        particle.setImageName(imageFile)
        outputParticlesSet.append(particle)
    return outputParticlesSet


def write_csv(filename, data):
    with open(filename, "w", newline="") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(data)


def read_boundingboxes(csv_file: str) -> Iterator[Tuple[Coordinate3D, float]]:
    with open(csv_file, "r") as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            coord = Coordinate3D()

            coord.setPosition(
                (float(row[1]) + float(row[4])) / 2,
                (float(row[2]) + float(row[5])) / 2,
                (float(row[3]) + float(row[6])) / 2,
            )
            w, h, d = (
                float(row[4]) - float(row[1]),
                float(row[5]) - float(row[2]),
                float(row[6]) - float(row[3]),
            )
            coord.setDim(w, h, d)
            yield coord, max(w, h, d)


def read_poses(poses_csv: str):
    with open(poses_csv, "r") as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            t = Transform()
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_euler(
                "XZX", [float(row[1]), float(row[2]), float(row[3])], degrees=True
            ).as_matrix()
            t.setMatrix(matrix)
            t.setShifts(float(row[4]), float(row[5]), float(row[6]))
            yield row[0], t


def save_boundingboxes(coords: SetOfCoordinates3D, csv_file: str):
    with open(csv_file, "w", newline="") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(
            ["index", "axis-1", "axis-2", "axis-3", "axis-1", "axis-2", "axis-3"]
        )
        for i, coord in enumerate(coords.iterCoordinates()):
            x, y, z = coord.getPosition()
            w, h, d = coord.getDim()
            csvwriter.writerow(
                [
                    i,
                    x - w / 2,
                    y - h / 2,
                    z - d / 2,
                    x + w / 2,
                    y + h / 2,
                    z + d / 2,
                ]
            )


def save_particles(
    particles_dir: str,
    particles: SetOfParticles,
    size: tuple[float, float, float],
    channel: int | None = None,
    voxel_size: tuple[float, float] | None = None,
):
    if not os.path.exists(particles_dir):
        os.makedirs(particles_dir, exist_ok=True)
    particles_paths: list[str] = []
    for particle in particles:
        particle: Particle
        im_name = particle.strId()
        im_newPath = os.path.join(particles_dir, im_name + ".ome.tiff")
        save_image(
            im_newPath, particle, size=size, channel=channel, voxel_size=voxel_size
        )
        particles_paths.append(im_newPath)

    return particles_paths


def _save_poses(
    poses_path: str,
    particles: SetOfParticles,
    particles_paths: list[str],
    prefix: str = "",
):
    mapping_particles_to_poses: dict[int, str] = {}
    with open(poses_path, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["index", "axis-1", "axis-2", "axis-3", "size"])
        for i, p, path in zip(range(len(particles_paths)), particles, particles_paths):
            p: Particle
            rotMat = p.getTransform().getRotationMatrix()
            euler_angles = list(
                map(
                    str,
                    Rotation.from_matrix(rotMat).as_euler("XZX", degrees=True).tolist(),
                )
            )
            trans = list(map(str, p.getTransform().getShifts().tolist()))
            mapping_particles_to_poses[p.getObjId()] = os.path.join(
                prefix, os.path.basename(path)
            )
            csvwriter.writerow(
                [os.path.join(prefix, os.path.basename(path))] + euler_angles + trans
            )
    return mapping_particles_to_poses


def save_particles_and_poses(
    rootdir: str,
    particles: SetOfParticles,
    size: tuple[float, float, float],
    channel: int | None = None,
    voxel_size: tuple[float, float] | None = None,
):
    particles_dir = os.path.join(rootdir, "particles")
    poses_path = os.path.join(rootdir, "poses.csv")
    os.makedirs(particles_dir, exist_ok=True)
    particles_paths = save_particles(
        os.path.join(rootdir, "particles"),
        particles,
        size,
        channel=channel,
        voxel_size=voxel_size,
    )
    mapping = _save_poses(poses_path, particles, particles_paths, prefix="particles")
    return particles_paths, particles_dir, poses_path, mapping


def save_image(
    new_path: str,
    image: Image,
    size: tuple[float, float, float],
    channel: int | None = None,
    voxel_size: tuple[float, float] | None = None,
):
    assert new_path.endswith(".ome.tiff")

    data = image.getData()
    if voxel_size and voxel_size != (original_voxel_size := image.getVoxelSize()):
        data = resample(
            data,
            (
                original_voxel_size[1] / voxel_size[1],
                original_voxel_size[0] / voxel_size[0],
                original_voxel_size[0] / voxel_size[0],
            ),
            multichannel=True,
            order=3,
        )

    data = interpolate_to_size(
        data,
        tuple(
            [
                math.ceil(s / vs)
                for s, vs in zip(size, (voxel_size[1], voxel_size[0], voxel_size[0]))
            ]
        ),
        multichannel=True,
        order=3,
    )

    if channel is not None:
        if channel > data.shape[0] - 1:
            channel = 0
        data = data[channel][None]

    Image.from_data(data, new_path, voxel_size=voxel_size)


def save_images(
    images: list[Image],
    output_dir: str,
    size: tuple[float, float, float],
    channel: int | None = None,
    voxel_size: tuple[float, float] | None = None,
):
    output_paths: list[str] = []
    for i, image in enumerate(images):
        im_newPath = os.path.join(
            output_dir, f"{image.__class__.__qualname__}-{i}.ome.tiff"
        )
        save_image(im_newPath, image, size, channel=channel, voxel_size=voxel_size)
        output_paths.append(im_newPath)
    return output_paths
