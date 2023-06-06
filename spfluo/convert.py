from typing import Dict, Iterator, List, Tuple
from spfluo.objects import SetOfParticles, Particle, Coordinate3D, Transform

import numpy as np
from scipy.spatial.transform import Rotation
import itertools
import csv
import glob
import os

from spfluo.objects.data import SetOfCoordinates3D


def getLastParticlesParams(folder):
    # Read rot_vecs
    print(os.path.join(folder, "estimated_rot_vecs_epoch_*.csv"))
    fpaths = glob.glob(os.path.join(folder, "estimated_rot_vecs_epoch_*.csv"))
    print(fpaths)
    rot_vecs_path = sorted(
        fpaths, key=lambda x: int(x.split("_")[-1][:-4]), reverse=True
    )[0]
    print(f"Opening {rot_vecs_path}")
    output = {}
    with open(rot_vecs_path, "r") as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            rot_vec = np.array(list(map(float, row[1:])))
            output[int(row[0])] = {
                "matrix": Rotation.from_rotvec(rot_vec, degrees=True).as_matrix()
            }
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
            am = np.array(particleParams["matrix"])
            matrix = np.row_stack((am, np.array([0, 0, 0])))
            matrix = np.column_stack((matrix, np.array([0, 0, 0, 1])))
            particle.setTransform(Transform(matrix))

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


def read_coordinate3D(csv_file: str) -> Iterator[Tuple[Coordinate3D, int]]:
    print(csv_file)
    with open(csv_file, "r") as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            coord = Coordinate3D()
            coord.setPosition(float(row[1]), float(row[2]), float(row[3]))
            yield coord, int(float(row[4]))


def save_coordinates3D(coords: SetOfCoordinates3D, csv_file: str):
    box_size = coords.getBoxSize()
    with open(csv_file, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["index", "axis-1", "axis-2", "axis-3", "size"])
        for i, coord in enumerate(coords.iterCoordinates()):
            x, y, z = coord.getPosition()
            csvwriter.writerow([i, x, y, z, box_size])
