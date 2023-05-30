from typing import Tuple
from spfluo.convert import read_coordinate3D
from spfluo.objects.data import FluoImage, SetOfCoordinates3D, SetOfFluoImages
from .protocol_base import ProtFluoPicking
from spfluo.viewers.view_picking import PickingView

from pyworkflow import BETA
from pyworkflow.protocol import Form
from pyworkflow.gui.dialog import askYesNo
from pyworkflow.utils.properties import Message
import pyworkflow.object as pwobj

import os


class ProtSPFluoPickingNapari(ProtFluoPicking):
    """
    Picking with the Napari plugin.
    """
    _label = 'manual picking'
    _devStatus = BETA

    def __init__(self, **kwargs):
        ProtFluoPicking.__init__(self, **kwargs)
    
    def _defineParams(self, form: Form):
        ProtFluoPicking._defineParams(self, form)
    
    def _insertAllSteps(self):
        self._insertFunctionStep(self.launchBoxingGUIStep, interactive=True)
    
    def getCsvPath(self, im: FluoImage) -> Tuple[str, str]:
        """ Get the FluoImage path and its csv file path"""
        path = im.getFileName()
        if path is None:
            raise Exception(f"{im} file path is None! Cannot launch napari.")
        path = os.path.abspath(path)
        fname, _ = os.path.splitext(os.path.basename(path))
        csv_file = fname + '.csv'
        csv_path = os.path.abspath(self._getExtraPath(csv_file))
        return path, csv_path
    
    def launchBoxingGUIStep(self):
        self.info_path = self._getExtraPath('info')

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
        sr_xy, sr_z = fluoimages.getSamplingRate()
        coords3D.setSamplingRate((sr_xy, sr_z))
        coords3D.enableAppend()
        box_size = None
        for imfluo in fluoimages.iterItems():
            # get csv filename
            _, csv_path = self.getCsvPath(imfluo)
            if os.path.exists(csv_path):
                for coord, box_size in read_coordinate3D(csv_path):
                    coord.setFluoImage(imfluo)
                    coord.setImageId(imfluo.getImgId())
                    coords3D.append(coord)
        if box_size:
            coords3D.setBoxSize(box_size)
        coords3D.write()
        
        name = self.OUTPUT_PREFIX + suffix
        self._defineOutputs(**{name: coords3D})
        self._defineRelation(pwobj.RELATION_SOURCE, fluoimages, coords3D)