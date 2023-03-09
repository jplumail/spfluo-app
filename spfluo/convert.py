from typing import Dict, List
from pwem.emlib.image import ImageHandler
from pwem.objects.data import Transform
import pyworkflow.utils as pwutils
from tomo.constants import TR_EMAN
from tomo.objects import SetOfSubTomograms, SubTomogram, Coordinate3D

import numpy as np
from scipy.spatial.transform import Rotation
import itertools
import tifffile
import csv
import glob
import os


def getLastParticlesParams(folder):
    # Read rot_vecs
    print(os.path.join(folder, 'estimated_rot_vecs_epoch_*.csv'))
    fpaths = glob.glob(os.path.join(folder, 'estimated_rot_vecs_epoch_*.csv'))
    print(fpaths)
    rot_vecs_path = sorted(fpaths, key=lambda x: int(x.split('_')[-1][:-4]), reverse=True)[0]
    print(f'Opening {rot_vecs_path}')
    output = {}
    with open(rot_vecs_path, 'r') as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            rot_vec = np.array(list(map(float, row[1:])))
            output[int(row[0])] = {'matrix': Rotation.from_rotvec(rot_vec, degrees=True).as_matrix()}    
    return output


def updateSetOfSubTomograms(
        inputSetOfSubTomograms: SetOfSubTomograms,
        outputSetOfSubTomograms: SetOfSubTomograms,
        particlesParams: Dict
    ) -> None:
    """Update a set of subtomograms from a template and copy attributes coverage/score/transform"""

    def updateSubTomogram(subTomogram: SubTomogram, index: int):
        particleParams = particlesParams.get(index)
        if not particleParams:
            print("Could not get params for particle %d" % index)
            setattr(subTomogram, "_appendItem", False)
        else:
            print("Got params for particle %d" % index)
            #setattr(subTomogram, EMAN_COVERAGE, Float(particleParams["coverage"]))
            #setattr(subTomogram, EMAN_SCORE, Float(particleParams["score"]))
            # Create 4x4 matrix from 4x3 e2spt_sgd align matrix and append row [0,0,0,1]
            am = np.array(particleParams["matrix"])
            # angles = numpy.array([am[0:3], am[4:7], am[8:11], [0, 0, 0]])
            # shift = numpy.array([am[3], am[7], am[11], 1])
            # matrix = numpy.column_stack((angles, shift.T))
            matrix = np.row_stack((am, np.array([0, 0, 0])))
            matrix = np.column_stack((matrix, np.array([0, 0, 0, 1])))
            subTomogram.setTransform(Transform(matrix), convention=None)

    outputSetOfSubTomograms.copyItems(inputSetOfSubTomograms,
                                      updateItemCallback=updateSubTomogram,
                                      itemDataIterator=itertools.count(0))
    outputSetOfSubTomograms.write()


def readSetOfSubTomograms(
        tomoFile: str,
        subtomoFileList: List[str],
        outputSubTomogramsSet: SetOfSubTomograms,
        coordSet: List[Coordinate3D]
    ) -> SetOfSubTomograms:
    for counter, subtomoFile in enumerate(subtomoFileList):
        print("Registering subtomogram %s - %s" % (counter, subtomoFile))
        subtomogram = SubTomogram()
        subtomogram.cleanObjId()
        subtomogram.setLocation(subtomoFile)
        subtomogram.setCoordinate3D(coordSet[counter])
        transformation = coordSet[counter]._eulerMatrix
        shift_x, shift_y, shift_z = transformation.getShifts()
        transformation.setShifts(shift_x,shift_y, shift_z)
        subtomogram.setTransform(transformation, convention=None)
        subtomogram.setVolName(tomoFile)
        outputSubTomogramsSet.append(subtomogram)
    return outputSubTomogramsSet

def convert_to_tif(mrc_filename, tiff_filename):
    ih = ImageHandler()
    img = ih.read(mrc_filename)
    data = img.getData()
    if data.ndim == 4:
        data = data[:,0]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got data of shape {data.shape}.")
    tifffile.imwrite(tiff_filename, data)


def write_csv(filename, data):
    with open(filename, 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(data)