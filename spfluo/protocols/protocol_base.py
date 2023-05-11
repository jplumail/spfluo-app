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
    EnumParam,
    PathParam,
    FloatParam,
    StringParam,
    BooleanParam,
    TupleParam,
    LEVEL_ADVANCED,
)
from pyworkflow.mapper.sqlite_db import SqliteDb
from pyworkflow.utils.properties import Message
from pyworkflow.protocol.params import Form, ElementGroup

import spfluo.objects as spobj
from spfluo.protocols.import_.base import ProtImportFiles, ProtImport

from typing import Any, Iterator, List, TypeVar, Type, Union


class ProtFluoBase:
    T = TypeVar("T", bound=Set)
    OUTPUT_PREFIX: str

    @classmethod
    def _createSet(cls, SetClass: Type[T], template, suffix, **kwargs) -> T:
        """Create a set and set the filename using the suffix.
        If the file exists, it will be deleted."""
        setFn = cls._getPath(template % suffix)
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
            indexes=["_imageId"],
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

    def iterOutputAttributes(self, cls: type) -> Iterator[str, Any]:
        ...

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
            "inputTomograms",
            PointerParam,
            label="Input Tomograms",
            important=True,
            pointerClass="SetOfTomograms",
            help="Select the Tomogram to be used during picking.",
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
    def _defineParams(self, form: Form) -> None:
        self._defineImportParams(form)

        self._defineAcquisitionParams(form)

    def _defineImportParams(self, form: Form) -> None:
        """Override to add options related to the different types
        of import that are allowed by each protocol.
        """
        importChoices = self._getImportChoices()

        form.addSection(label="Import")
        if len(importChoices) > 1:  # not only from files
            form.addParam(
                "importFrom",
                EnumParam,
                choices=importChoices,
                default=self._getDefaultChoice(),
                label="Import from",
                help="Select the type of import.",
            )
        else:
            form.addHidden(
                "importFrom",
                EnumParam,
                choices=importChoices,
                default=self.IMPORT_FROM_FILES,
                label="Import from",
                help="Select the type of import.",
            )

        form.addParam(
            "filesPath",
            PathParam,
            label="Files directory",
            help="Directory with the files you want to import.\n\n"
            "The path can also contain wildcards to select"
            "from several folders. \n\n"
            "Examples:\n"
            "  ~/Images/data/day??_img/\n"
            "Each '?' represents one unknown character\n\n"
            "  ~/Images/data/day*_images/\n"
            "'*' represents any number of unknown characters\n\n"
            "  ~/Images/data/day#_images/\n"
            "'#' represents one digit that will be used as "
            "image ID\n\n"
            "NOTE: wildcard characters ('*', '?', '#') "
            "cannot appear in the actual path.)",
        )
        form.addParam(
            "filesPattern",
            StringParam,
            label="Pattern",
            help="Pattern of the files to be imported.\n\n"
            "The pattern can contain standard wildcards such as\n"
            "*, ?, etc, or special ones like ### to mark some\n"
            "digits in the filename as ID.\n\n"
            "NOTE: wildcards and special characters "
            "('*', '?', '#', ':', '%') cannot appear in the "
            "actual path.",
        )
        form.addParam(
            "copyFiles",
            BooleanParam,
            default=False,
            expertLevel=LEVEL_ADVANCED,
            label="Copy files?",
            help="By default the files are not copied into the "
            "project to avoid data duplication and to save "
            "disk space. Instead of copying, symbolic links are "
            "created pointing to original files. This approach "
            "has the drawback that if the project is moved to "
            "another computer, the links need to be restored.",
        )

    def _defineAcquisitionParams(self, form: Form) -> None:
        """Override to add options related to acquisition info."""
        form.addGroup("Sampling rate")
        form.addParam("sr_xy", FloatParam, label="XY")
        form.addParam("sr_z", FloatParam, label="Z")

    def _validate(self):
        pass


class ProtFluoParticleAveraging(Protocol, ProtFluoBase):
    """Base class for subtomogram averaging protocols."""

    pass
