# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (delarosatrevin@scilifelab.se) [1]
# *
# * [1] SciLifeLab, Stockholm University
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

import pyworkflow as pw
from pyworkflow.object import Set
from pyworkflow.protocol import Protocol
from pyworkflow.protocol.params import (
    PointerParam,
    FloatParam,
)
from pyworkflow.mapper.sqlite_db import SqliteDb
from pyworkflow.utils.properties import Message
from pyworkflow.protocol.params import Form

import spfluo.objects as spobj
from spfluo.protocols.import_.base import ProtImportFiles, ProtImportFile, ProtImport

from typing import List, TypeVar, Type


class ProtFluoBase:
    T = TypeVar("T", bound=Set)
    OUTPUT_PREFIX: str

    def _createSet(self, SetClass: Type[T], template, suffix, **kwargs) -> T:
        """Create a set and set the filename using the suffix.
        If the file exists, it will be deleted."""
        setFn = self._getPath(template % suffix)
        # Close the connection to the database if
        # it is open before deleting the file
        pw.utils.cleanPath(setFn)

        SqliteDb.closeConnection(setFn)
        setObj = SetClass(filename=setFn, **kwargs)
        return setObj

    def _createSetOfCoordinates3D(
        self, volSet: spobj.SetOfFluoImages, suffix: str = ""
    ) -> spobj.SetOfCoordinates3D:
        coord3DSet: spobj.SetOfCoordinates3D = self._createSet(
            spobj.SetOfCoordinates3D,
            "coordinates%s.sqlite",
            suffix,
            indexes=[spobj.Coordinate3D.IMAGE_ID_ATTR],
        )
        coord3DSet.setPrecedents(volSet)
        return coord3DSet

    def _createSetOfFluoImages(self, suffix: str = "") -> spobj.SetOfFluoImages:
        return self._createSet(spobj.SetOfFluoImages, "fluoimages%s.sqlite", suffix)

    def _createSetOfParticles(self, suffix: str = "") -> spobj.SetOfParticles:
        return self._createSet(spobj.SetOfParticles, "particles%s.sqlite", suffix)

    # def _createSetOfAverageSubTomograms(self, suffix='')-> spobj.SetOfAverageSubTomograms:
    #    return self._createSet(spobj.SetOfAverageSubTomograms,
    #                           'avgSubtomograms%s.sqlite', suffix)

    # def _createSetOfClassesSubTomograms(self, subTomograms, suffix='')->spobj.SetOfClassesSubTomograms:
    #    classes = self._createSet(spobj.SetOfClassesSubTomograms,
    #                              'subtomogramClasses%s.sqlite', suffix)
    #    classes.setImages(subTomograms)
    #    return classes

    # def _createSetOfLandmarkModels(self, suffix='') -> spobj.SetOfLandmarkModels:
    #    return self._createSet(spobj.SetOfLandmarkModels, 'setOfLandmarks%s.sqlite', suffix)

    # def _createSetOfMeshes(self, volSet, suffix='')->spobj.SetOfMeshes:
    #    meshSet = self._createSet(spobj.SetOfMeshes,
    #                              'meshes%s.sqlite', suffix)
    #    meshSet.setPrecedents(volSet)
    #    return meshSet

    def _getOutputSuffix(self, cls: type) -> str:
        """Get the name to be used for a new output.
        For example: output3DCoordinates7.
        It should take into account previous outputs
        and number with a higher value.
        """
        maxCounter = -1
        for attrName, _ in self.iterOutputAttributes(cls):
            suffix = attrName.replace(self.OUTPUT_PREFIX, "")
            try:
                counter = int(suffix)
            except:
                counter = 1  # when there is not number assume 1
            maxCounter = max(counter, maxCounter)

        return str(maxCounter + 1) if maxCounter > 0 else ""  # empty if not output


class ProtFluoPicking(ProtImport, ProtFluoBase):
    OUTPUT_PREFIX = "output3DCoordinates"

    """ Base class for Fluo boxing protocols. """

    def _defineParams(self, form: Form) -> None:
        form.addSection(label="Input")
        form.addParam(
            "inputFluoImages",
            PointerParam,
            label="Input Images",
            important=True,
            pointerClass="SetOfFluoImages",
            help="Select the Image to be used during picking.",
        )

    def _summary(self) -> List[str]:
        summary = []
        if self.isFinished() and self.getOutputsSize() >= 1:
            for key, output in self.iterOutputAttributes():
                summary.append("*%s:*\n%s" % (key, output.getSummary()))
        else:
            summary.append(Message.TEXT_NO_OUTPUT_CO)
        return summary


class ProtFluoImportFiles(ProtImportFiles, ProtFluoBase):
    def _defineAcquisitionParams(self, form: Form) -> None:
        """Override to add options related to acquisition info."""
        form.addGroup("Voxel size")
        form.addParam("vs_xy", FloatParam, label="XY")
        form.addParam("vs_z", FloatParam, label="Z")

    def _validate(self):
        pass


class ProtFluoImportFile(
    ProtImportFile, ProtFluoBase
):  # TODO: find a better architecture
    def _defineAcquisitionParams(self, form: Form) -> None:
        """Override to add options related to acquisition info."""
        form.addGroup("Voxel size")
        form.addParam("vs_xy", FloatParam, label="XY")
        form.addParam("vs_z", FloatParam, label="Z")

    def _validate(self):
        pass


class ProtFluoParticleAveraging(Protocol, ProtFluoBase):
    """Base class for subtomogram averaging protocols."""

    pass
