from pwem.objects.data import Volume
from pwem.constants import (NO_INDEX, ALIGN_NONE, ALIGN_2D, ALIGN_3D,
                            ALIGN_PROJ, ALIGNMENTS)
from pyworkflow.object import Integer, Float, String, Pointer, Boolean, CsvList, Object, Scalar, Set

import os
import numpy as np
import json


class Score(Object):

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)
        self._score = Float()
        self._coordPointer = Pointer(objDoStore=False)
        self._coordId = Integer()
    
    def setCoord(self, coord):
        self._coordPointer.set(coord)
        self._coordId.set(coord.getObjId())
    
    def getCoord(self):
        return self._coordPointer.get()
    
    def setScore(self, score):
        self._score.set(score)
    
    def getScore(self):
        return self._score

class ScoreSet(Set):
    ITEM_TYPE = Score

    def __init__(self, **kwargs):
       Set.__init__(self, **kwargs)

class Matrix(Scalar):
    def __init__(self, **kwargs):
        Scalar.__init__(self, **kwargs)
        self._matrix = np.eye(4)

    def _convertValue(self, value):
        """Value should be a str with comma separated values
        or a list.
        """
        self._matrix = np.array(json.loads(value))

    def getObjValue(self):
        self._objValue = json.dumps(self._matrix.tolist())
        return self._objValue

    def setValue(self, i, j, value):
        self._matrix[i, j] = value

    def getMatrix(self):
        """ Return internal numpy matrix. """
        return self._matrix

    def setMatrix(self, matrix):
        """ Override internal numpy matrix. """
        self._matrix = matrix

    def __str__(self):
        return np.array_str(self._matrix)

    def _copy(self, other, copyDict, copyId, level=1, ignoreAttrs=[]):
        """ Override the default behaviour of copy
        to also copy array data.
        """
        self.setMatrix(np.copy(other.getMatrix()))
        self._objValue = other._objValue


class Transform(Object):
    """ This class will contain a transformation matrix
    that can be applied to 2D/3D objects like images and volumes.
    It should contain information about euler angles, translation(or shift)
    and mirroring.
    Shifts are stored in pixels as treated in extract coordinates, or assign angles,...
    """

    # Basic Transformation factory
    ROT_X_90_CLOCKWISE = 'rotX90c'
    ROT_Y_90_CLOCKWISE = 'rotY90c'
    ROT_Z_90_CLOCKWISE = 'rotZ90c'
    ROT_X_90_COUNTERCLOCKWISE = 'rotX90cc'
    ROT_Y_90_COUNTERCLOCKWISE = 'rotY90cc'
    ROT_Z_90_COUNTERCLOCKWISE = 'rotZ90cc'

    def __init__(self, matrix=None, **kwargs):
        Object.__init__(self, **kwargs)
        self._matrix = Matrix()
        if matrix is not None:
            self.setMatrix(matrix)

    def getMatrix(self):
        return self._matrix.getMatrix()

    def getRotationMatrix(self):
        M = self.getMatrix()
        return M[:3, :3]

    def getShifts(self):
        M = self.getMatrix()
        return M[1, 4], M[2, 4], M[3, 4]

    def getMatrixAsList(self):
        """ Return the values of the Matrix as a list. """
        return self._matrix.getMatrix().flatten().tolist()

    def setMatrix(self, matrix):
        self._matrix.setMatrix(matrix)

    def __str__(self):
        return str(self._matrix)

    def scale(self, factor):
        m = self.getMatrix()
        m *= factor
        m[3, 3] = 1.

    def scaleShifts(self, factor):
        # By default Scipion uses a coordinate system associated with the volume rather than the projection
        m = self.getMatrix()
        m[0, 3] *= factor
        m[1, 3] *= factor
        m[2, 3] *= factor

    def invert(self):
        self._matrix(np.linalg.inv(self._matrix.getMatrix()))
        return self._matrix

    def getShifts(self):
        m = self.getMatrix()
        return m[0, 3], m[1, 3], m[2, 3]

    def setShifts(self, x, y, z):
        m = self.getMatrix()
        m[0, 3] = x
        m[1, 3] = y
        m[2, 3] = z

    def setShiftsTuple(self, shifts):
        self.setShifts(shifts[0], shifts[1], shifts[2])

    def composeTransform(self, matrix):
        """Apply a transformation matrix to the current matrix """
        new_matrix = np.matmul(matrix, self.getMatrix())
        # new_matrix = matrix * self.getMatrix()
        self._matrix.setMatrix(new_matrix)

    @classmethod
    def create(cls, type):
        if type == cls.ROT_X_90_CLOCKWISE:
            return Transform(matrix=np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1]]))
        elif type == cls.ROT_X_90_COUNTERCLOCKWISE:
            return Transform(matrix=np.array([
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]]))
        elif type == cls.ROT_Y_90_CLOCKWISE:
            return Transform(matrix=np.array([
                [1, 0, -1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]]))
        elif type == cls.ROT_Y_90_COUNTERCLOCKWISE:
            return Transform(matrix=np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]]))
        elif type == cls.ROT_Z_90_CLOCKWISE:
            return Transform(matrix=np.array([
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]))
        elif type == cls.ROT_Z_90_COUNTERCLOCKWISE:
            return Transform(matrix=np.array([
                [0, -1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]))
        else:
            TRANSFORMATION_FACTORY_TYPES = [
                cls.ROT_X_90_CLOCKWISE,
                cls.ROT_Y_90_CLOCKWISE,
                cls.ROT_Z_90_CLOCKWISE,
                cls.ROT_X_90_COUNTERCLOCKWISE,
                cls.ROT_Y_90_COUNTERCLOCKWISE,
                cls.ROT_Z_90_COUNTERCLOCKWISE
            ]
            raise Exception('Introduced Transformation type is not recognized.\nAdmitted values are\n'
                            '%s' % ' '.join(TRANSFORMATION_FACTORY_TYPES))


class Image(Object):
    """Represents a Fluo image object"""

    def __init__(self, location=None, **kwargs):
        """
         Params:
        :param location: Could be a valid location: (index, filename)
        or  filename
        """
        Object.__init__(self, **kwargs)
        # Image location is composed by an index and a filename
        self._index = Integer(0)
        self._filename = String()
        self._samplingRate = Float()
        self._psfModel = None
        self._acquisition = None
        # _transform property will store the transformation matrix
        # this matrix can be used for 2D/3D alignment or
        # to represent projection directions
        self._transform = None
        # default origin by default is box center =
        # (Xdim/2, Ydim/2,Zdim/2)*sampling
        # origin stores a matrix that using as input the point (0,0,0)
        # provides  the position of the actual origin in the system of
        # coordinates of the default origin.
        # _origin is an object of the class Transform shifts
        # units are A.
        # Origin coordinates follow the MRC convention
        self._origin = None
        if location:
            self.setLocation(location)

    def getSamplingRate(self):
        """ Return image sampling rate. (A/pix) """
        return self._samplingRate.get()

    def setSamplingRate(self, sampling):
        self._samplingRate.set(sampling)

    def getFormat(self):
        pass

    def getDataType(self):
        pass

    def getDimensions(self):
        """getDimensions is redundant here but not in setOfVolumes
         create it makes easier to create protocols for both images
         and sets of images
        """
        return self.getDim()

    def getDim(self):
        """Return image dimensions as tuple: (Xdim, Ydim, Zdim)"""
        from pwem.emlib.image import ImageHandler
        x, y, z, n = ImageHandler().getDimensions(self)
        return None if x is None else (x, y, z)

    def getXDim(self):
        return self.getDim()[0] if self.getDim() is not None else 0

    def getYDim(self):
        return self.getDim()[1] if self.getDim() is not None else 0

    def getIndex(self):
        return self._index.get()

    def setIndex(self, index):
        self._index.set(index)

    def getFileName(self):
        """ Use the _objValue attribute to store filename. """
        return self._filename.get()

    def setFileName(self, filename):
        """ Use the _objValue attribute to store filename. """
        self._filename.set(filename)

    def getLocation(self):
        """ This function return the image index and filename.
        It will only differs from getFileName, when the image
        is contained in a stack and the index make sense.
        """
        return self.getIndex(), self.getFileName()

    def setLocation(self, *args):
        """ Set the image location, see getLocation.
        Params:
            First argument can be:
             1. a tuple with (index, filename)
             2. a index, this implies a second argument with filename
             3. a filename, this implies index=NO_INDEX
        """
        first = args[0]
        t = type(first)
        if t == tuple:
            index, filename = first
        elif t == int:
            index, filename = first, args[1]
        elif t == str:
            index, filename = NO_INDEX, first
        else:
            raise Exception('setLocation: unsupported type %s as input.' % t)

        self.setIndex(index)
        self.setFileName(filename)

    def getBaseName(self):
        return os.path.basename(self.getFileName())

    def copyInfo(self, other):
        """ Copy basic information """
        self.copyAttributes(other, '_samplingRate')

    def copyLocation(self, other):
        """ Copy location index and filename from other image. """
        self.setIndex(other.getIndex())
        self.setFileName(other.getFileName())

    def hasPSF(self):
        return self._psfModel is not None

    def getPSF(self):
        """ Return the CTF model """
        return self._psfModel

    def setCTF(self, newPSF):
        self._psfModel = newPSF

    def hasAcquisition(self):
        return self._acquisition is not None

    def getAcquisition(self):
        return self._acquisition

    def setAcquisition(self, acquisition):
        self._acquisition = acquisition

    def hasTransform(self):
        return self._transform is not None

    def getTransform(self)-> Transform:
        return self._transform

    def setTransform(self, newTransform):
        self._transform = newTransform

    def hasOrigin(self):
        return self._origin is not None

    def getOrigin(self, force=False):
        """shifts in A"""
        if self.hasOrigin():
            return self._origin
        else:
            if force:
                return self._getDefaultOrigin()
            else:
                return None

    def _getDefaultOrigin(self):
        sampling = self.getSamplingRate()
        t = Transform()
        x, y, z = self.getDim()
        if z > 1:
            z = z / -2.
        t.setShifts(x / -2. * sampling, y / -2. * sampling, z * sampling)
        return t  # The identity matrix

    def getShiftsFromOrigin(self):
        origin = self.getOrigin(force=True).getShifts()
        x = origin[0]
        y = origin[1]
        z = origin[2]
        return x, y, z
        # x, y, z are floats in Angstroms

    def setShiftsInOrigin(self, x, y, z):
        origin = self.getOrigin(force=True)
        origin.setShifts(x, y, z)

    def setOrigin(self, newOrigin=None):
        """If None, default origin will be set.
        Note: shifts are in Angstroms"""
        if newOrigin:
            self._origin = newOrigin
        else:
            self._origin = self._getDefaultOrigin()

    def originResampled(self, originNotResampled, oldSampling):
        factor = self.getSamplingRate() / oldSampling
        shifts = originNotResampled.getShifts()
        origin = self.getOrigin(force=True)
        origin.setShifts(shifts[0] * factor,
                         shifts[1] * factor,
                         shifts[2] * factor)
        return origin

    def __str__(self):
        """ String representation of an Image. """
        dim = self.getDim()
        dimStr = str(ImageDim(*dim)) if dim else 'No-Dim'
        return ("%s (%s, %0.2f Å/px)" % (self.getClassName(), dimStr,
                                         self.getSamplingRate() or 99999.))

    def getFiles(self):
        filePaths = set()
        filePaths.add(self.getFileName())
        return filePaths

    def setMRCSamplingRate(self):
        """ Sets the sampling rate to the mrc file represented by this image"""
        from pwem.convert.headers import setMRCSamplingRate
        setMRCSamplingRate(self.getFileName(), self.getSamplingRate())

class Volume():
    def __init__(self, **kwargs):
        Volume.__init__(self, **kwargs)
        self._acquisition = None
        self._tsId = String(kwargs.get('tsId', None))
        self._dim = None

    def getTsId(self):
        """ Get unique TiltSeries ID, usually retrieved from the
        file pattern provided by the user at the import time.
        """
        return self._tsId.get()

    def setTsId(self, value):
        self._tsId.set(value)

    def getAcquisition(self):
        return self._acquisition

    def setAcquisition(self, acquisition):
        self._acquisition = acquisition

    def hasAcquisition(self):
        return (self._acquisition is not None
                and self._acquisition.getAngleMin() is not None
                and self._acquisition.getAngleMax() is not None)

    def getDim(self):
        """Return image dimensions as tuple: (Xdim, Ydim, Zdim)"""
        if self._dim is None:
            from pwem.emlib.image import ImageHandler

            fn = self.getFileName()
            if fn is not None and os.path.exists(fn.replace(':mrc', '')):
                x, y, z, n = ImageHandler().getDimensions(self)

                # Some volumes in mrc format can have the z dimension
                # as n dimension, so we need to consider this case.
                if z > 1:
                    self._dim = (x, y, z)
                    return x, y, z
                else:
                    self._dim = (x, y, n)
                    return x, y, n
        else:
            return self._dim
        return None

    def copyInfo(self, other):
        """ Copy basic information """
        super().copyInfo(other)
        self.copyAttributes(other, '_acquisition', self.TS_ID_FIELD)
        if other.hasOrigin():
            self.copyAttributes(other, '_origin')