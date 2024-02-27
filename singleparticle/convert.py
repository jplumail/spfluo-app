import csv
import glob
import itertools
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


def getLastParticlesParams(folder):
    # Read poses
    print(os.path.join(folder, "estimated_poses_epoch_*.csv"))
    fpaths = glob.glob(os.path.join(folder, "estimated_poses_epoch_*.csv"))
    print(fpaths)
    poses_path = sorted(fpaths, key=lambda x: int(x.split("_")[-1][:-4]), reverse=True)[
        0
    ]
    print(f"Opening {poses_path}")
    output = {}
    with open(poses_path, "r") as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            rot = np.array(list(map(float, row[1:4])))
            trans = np.array(list(map(float, row[4:7])))
            H = np.zeros((4, 4), dtype=float)
            H[:3, :3] = Rotation.from_euler("XZX", rot, degrees=True).as_matrix()
            H[:3, 3] = trans
            output[int(row[0])] = {"homogeneous_transform": H}
    return output


def updateSetOfParticles(
    inputSetOfParticles: SetOfParticles,
    outputSetOfParticles: SetOfParticles,
    particlesParams: Dict,
) -> None:
    """Update a set of particles from a template
    and copy attributes coverage/score/transform"""

    def updateParticle(particle: Particle, index: int):
        particleParams = particlesParams.get(index)
        if not particleParams:
            print("Could not get params for particle %d" % index)
            setattr(particle, "_appendItem", False)
        else:
            print("Got params for particle %d" % index)
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
        print("Registering particle %s - %s" % (counter, particleFile))
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
    with open(filename, "w") as f:
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
            yield t, row[0]


def save_translations(coords: SetOfCoordinates3D, csv_file: str):
    box_size = coords.getBoxSize()
    with open(csv_file, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["index", "axis-1", "axis-2", "axis-3", "size"])
        for i, coord in enumerate(coords.iterCoordinates()):
            x, y, z = coord.getPosition()
            csvwriter.writerow([i, x, y, z, box_size])


def save_particles(
    particles_dir: str, particles: SetOfParticles, channel: int = None
) -> Tuple[List[str], int]:
    print("Creating particles directory")
    if not os.path.exists(particles_dir):
        os.makedirs(particles_dir, exist_ok=True)
    particles_paths = []
    max_dim = 0
    for im in particles:
        im: Particle
        max_dim = max(max_dim, max(im.getDim()))
        im_path = os.path.abspath(im.getFileName())
        ext = os.path.splitext(im_path)[1]
        im_name = im.strId()
        im_newPath = os.path.join(particles_dir, im_name + ".ome.tiff")
        particles_paths.append(im_newPath)
        if im.getNumChannels() > 1 and channel is not None:
            Particle.from_data(
                im.getData()[channel][None], im_newPath, voxel_size=im.getVoxelSize()
            )
        else:
            if ext.endswith(".tiff") or ext.endswith(".tif"):
                os.link(im_path, im_newPath)
            else:
                raise NotImplementedError(
                    f"Found ext {ext} in particles: {im_path}."
                    "Only tiff file are supported."
                )

    return particles_paths, max_dim


def save_poses(path: str, particles: SetOfParticles):
    with open(path, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["index", "axis-1", "axis-2", "axis-3", "size"])
        for p in particles:
            p: Particle
            rotMat = p.getTransform().getRotationMatrix()
            euler_angles = list(
                map(
                    str,
                    Rotation.from_matrix(rotMat).as_euler("XZX", degrees=True).tolist(),
                )
            )
            trans = list(map(str, p.getTransform().getShifts().tolist()))
            csvwriter.writerow([str(p.getObjId())] + euler_angles + trans)


def save_particles_and_poses(
    root_dir: str, particles: SetOfParticles, channel: int = None
) -> Tuple[List[str], int]:
    print("Creating particles directory")
    particles_dir = os.path.join(root_dir, "particles")
    csv_path = os.path.join(root_dir, "poses.csv")
    if not os.path.exists(root_dir):
        os.makedirs(particles_dir, exist_ok=True)
    particles_paths = []
    max_dim = 0
    with open(csv_path, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["index", "axis-1", "axis-2", "axis-3", "size"])

        for im in particles:
            im: Particle
            max_dim = max(max_dim, max(im.getDim()))
            im_path = os.path.abspath(im.getFileName())
            ext = os.path.splitext(im_path)[1]
            im_name = im.strId()
            im_newPath = os.path.join(particles_dir, im_name + ".ome.tiff")
            particles_paths.append(im_newPath)
            if channel is not None and im.getNumChannels() > 1:
                Particle.from_data(
                    im.getData()[channel][None],
                    im_newPath,
                    voxel_size=im.getVoxelSize(),
                )
            else:
                if ext.endswith(".tiff") or ext.endswith(".tif"):
                    os.link(im_path, im_newPath)
                else:
                    raise NotImplementedError(
                        f"Found ext {ext} in particles: {im_path}."
                        "Only tiff file are supported."
                    )

            # Write pose
            rotMat = im.getTransform().getRotationMatrix()
            euler_angles = list(
                map(
                    str,
                    Rotation.from_matrix(rotMat).as_euler("XZX", degrees=True).tolist(),
                )
            )
            trans = list(map(str, im.getTransform().getShifts().tolist()))
            csvwriter.writerow(
                [os.path.join("particles", im_name + ".ome.tiff")]
                + euler_angles
                + trans
            )

    return particles_paths, max_dim


def save_image(new_path: str, image: Image, channel: int = None):
    image_path = os.path.abspath(image.getFileName())
    ext = os.path.splitext(image_path)[1]
    if image.getNumChannels() > 1 and channel is not None:
        Particle.from_data(
            image.getData()[channel][None], new_path, voxel_size=image.getVoxelSize()
        )
    else:
        if ext.endswith(".tiff") or ext.endswith(".tif"):
            os.link(image_path, new_path)
        else:
            raise NotImplementedError(
                f"Found ext {ext} in particles: {image}."
                "Only tiff file are supported."
            )
