# coding=utf-8

from os.path import abspath, basename
import re
from typing import List, Optional, Tuple

from pwem.convert.headers import Ccp4Header
from spfluo.objects import Transform
from pyworkflow import BETA
from pyworkflow import utils as pwutils
from pyworkflow.utils.path import createAbsLink, removeExt
import pyworkflow.protocol.params as params

from .protocol_base import ProtFluoImportFiles
from ..objects import FluoImage, SetOfFluoImages


def _getUniqueFileName(pattern, filename, filePaths=None):
    if filePaths is None:
        filePaths = [re.split(r"[$*#?]", pattern)[0]]

    commPath = pwutils.commonPath(filePaths)
    return filename.replace(commPath + "/", "").replace("/", "_")


OUTPUT_NAME = "FluoImages"


class ProtImportFluoImages(ProtFluoImportFiles):
    """Protocol to import a set of fluoimages to the project"""

    _outputClassName = "SetOfFluoImages"
    _label = "import fluoimages"
    _devStatus = BETA
    _possibleOutputs = {OUTPUT_NAME: SetOfFluoImages}

    def __init__(self, **args):
        ProtFluoImportFiles.__init__(self, **args)
        self.FluoImages = None

    def _defineParams(self, form):
        ProtFluoImportFiles._defineParams(self, form)

    def _getImportChoices(self):
        """Return a list of possible choices
        from which the import can be done.
        """
        return ["eman"]

    def _insertAllSteps(self):
        self._insertFunctionStep(
            "importFluoImagesStep",
            self.getPattern(),
            (self.sr_xy.get(), self.sr_z.get()),
        )

    # --------------------------- STEPS functions -----------------------------

    def importFluoImagesStep(
        self, pattern: str, samplingRate: Tuple[float, float]
    ) -> None:
        """Copy images matching the filename pattern
        Register other parameters.
        """
        self.info("Using pattern: '%s'" % pattern)

        imgSet = self._createSetOfFluoImages()
        imgSet.setSamplingRate(samplingRate)

        fileNameList = []
        for fileName, fileId in self.iterFiles():
            img = FluoImage(filename=fileName)
            img.setSamplingRate(samplingRate)

            # Set default origin
            origin = Transform()
            dim = img.getDim()
            if dim is None:
                raise ValueError("Image '%s' has no dimension" % fileName)
            x, y, z = dim
            origin.setShifts(
                x / -2.0 * samplingRate,
                y / -2.0 * samplingRate,
                z / -2.0 * samplingRate,
            )
            img.setOrigin(origin)

            newFileName = basename(fileName).split(":")[0]
            if newFileName in fileNameList:
                newFileName = _getUniqueFileName(
                    self.getPattern(), fileName.split(":")[0]
                )

            fileNameList.append(newFileName)

            imgId = removeExt(newFileName)
            img.setImgId(imgId)

            createAbsLink(abspath(fileName), abspath(self._getExtraPath(newFileName)))

            img.cleanObjId()
            img.setFileName(self._getExtraPath(newFileName))
            imgSet.append(img)

        self._defineOutputs(**{OUTPUT_NAME: imgSet})

    # --------------------------- INFO functions ------------------------------
    def _hasOutput(self) -> bool:
        return self.FluoImages is not None

    def _getTomMessage(self) -> str:
        return "FluoImages %s" % self.getObjectTag(OUTPUT_NAME)

    def _summary(self) -> str:
        try:
            summary = []
            if self._hasOutput():
                summary.append(
                    "%s imported from:\n%s" % (self._getTomMessage(), self.getPattern())
                )

                if self.samplingRate.get():
                    summary.append(
                        "Sampling rate: *%0.2f* (Å/px)" % self.samplingRate.get()
                    )

                x, y, z = self.FluoImages.getFirstItem().getShiftsFromOrigin()
                summary.append(
                    "FluoImages Origin (x,y,z):\n"
                    "    x: *%0.2f* (Å/px)\n"
                    "    y: *%0.2f* (Å/px)\n"
                    "    z: *%0.2f* (Å/px)" % (x, y, z)
                )

        except Exception as e:
            print(e)

        return summary

    def _methods(self) -> List[str]:
        methods = []
        if self._hasOutput():
            methods.append(
                " %s imported with a sampling rate *%0.2f*"
                % (self._getTomMessage(), self.samplingRate.get()),
            )
        return methods

    def _getVolumeFileName(self, fileName: str, extension: Optional[str] = None) -> str:
        if extension is not None:
            baseFileName = (
                "import_" + str(basename(fileName)).split(".")[0] + ".%s" % extension
            )
        else:
            baseFileName = "import_" + str(basename(fileName)).split(":")[0]

        return self._getExtraPath(baseFileName)

    def _validate(self) -> List[str]:
        errors = []
        try:
            next(self.iterFiles())
        except StopIteration:
            errors.append(
                "No files matching the pattern %s were found." % self.getPattern()
            )
        return errors
