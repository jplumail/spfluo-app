import os
from typing import Tuple

import pyworkflow.object as pwobj
from pwfluo.objects import FluoImage, SetOfCoordinates3D, SetOfFluoImages
from pwfluo.protocols import ProtFluoPicking
from pyworkflow import BETA
from pyworkflow.gui.dialog import askYesNo
from pyworkflow.protocol import Form
from pyworkflow.utils.properties import Message

from singleparticle.convert import read_boundingboxes
from singleparticle.viewers.view_picking import PickingView


class ProtSingleParticlePickingNapari(ProtFluoPicking):
    """
    Picking with the Napari plugin.
    """

    _label = "manual picking"
    _devStatus = BETA

    def __init__(self, **kwargs):
        ProtFluoPicking.__init__(self, **kwargs)

    def _defineParams(self, form: Form):
        ProtFluoPicking._defineParams(self, form)

    def _insertAllSteps(self):
        self._insertFunctionStep(self.launchBoxingGUIStep, interactive=True)

    def getCsvPath(self, im: FluoImage) -> Tuple[str, str]:
        """Get the FluoImage path and its csv file path"""
        path = im.getFileName()
        if path is None:
            raise Exception(f"{im} file path is None! Cannot launch napari.")
        path = os.path.abspath(path)
        fname, _ = os.path.splitext(os.path.basename(path))
        csv_file = fname + ".csv"
        csv_path = os.path.abspath(self._getExtraPath(csv_file))
        return path, csv_path

    def launchBoxingGUIStep(self):
        self.info_path = self._getExtraPath("info")

        fluoList = []
        fluoimages: SetOfFluoImages = self.inputFluoImages.get()
        for i, fluo in enumerate(fluoimages.iterItems()):
            fluo: FluoImage = fluo
            fluoImage = fluo.clone()
            fluoImage.count = 0
            fluoImage.in_viewer = False
            fluoList.append(fluoImage)

        view = PickingView(None, self, fluoList)
        view.show()

        # Open dialog to request confirmation to create output
        import tkinter as tk

        frame = tk.Frame()
        if askYesNo(Message.TITLE_SAVE_OUTPUT, Message.LABEL_SAVE_OUTPUT, frame):
            self.createOuput()

    def createOuput(self):
        fluoimages: SetOfFluoImages = self.inputFluoImages.get()
        suffix = self._getOutputSuffix(SetOfCoordinates3D)

        coords3D = self._createSetOfCoordinates3D(fluoimages, suffix)
        coords3D.setName("fluoCoord")
        vs_xy, vs_z = fluoimages.getVoxelSize()
        coords3D.setVoxelSize((vs_xy, vs_z))
        coords3D.enableAppend()
        max_box_size = None
        for imfluo in fluoimages.iterItems():
            # get csv filename
            _, csv_path = self.getCsvPath(imfluo)
            if os.path.exists(csv_path):
                for coord, box_size in read_boundingboxes(csv_path):
                    coord.setFluoImage(imfluo)
                    coord.setImageId(imfluo.getImgId())
                    coords3D.append(coord)
                    if (not max_box_size) or box_size > max_box_size:
                        max_box_size = box_size
        coords3D.write()

        name = self.OUTPUT_PREFIX + suffix
        self._defineOutputs(**{name: coords3D})
        self._defineRelation(pwobj.RELATION_SOURCE, fluoimages, coords3D)
