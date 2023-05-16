from .protocol_base import ProtFluoPicking
from spfluo import Plugin
from spfluo.viewers.views_tkinter_tree import FluoImagesTreeProvider, NapariDialog

from pyworkflow import BETA
from pyworkflow.protocol import Protocol, params, Form
from pyworkflow.gui.dialog import askYesNo
from pyworkflow.utils.properties import Message

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
        self._insertFunctionStep(self.launchBoxingGUIStep)
    
    def launchBoxingGUIStep(self):
        self.info_path = self._getExtraPath('info')
        lastOutput = None
        # Should get last outputs and put it in lastOuput variable ?
        #if self.getOutputsSize() > 0:
        #    pwutils.makePath(self.info_path)
        #    self.json_files, self.tomo_files = jsonFilesFromSet(self.inputTomograms.get(), self.info_path)
        #    lastOutput = [output for _, output in self.iterOutputAttributes()][-1]
        #    _ = setCoords3D2Jsons(self.json_files, lastOutput)
        #    pass
        
        # get the number of annotated things in lastOutput
        #if lastOutput is not None:
        #    volIds = lastOutput.aggregate(["MAX", "COUNT"], "_volId", ["_volId"])
        #    volIds = dict([(d['_volId'], d["COUNT"]) for d in volIds])
        #else:
        #    volIds = dict()

        fluoList = []
        for fluo in self.inputFluoImages.get().iterItems():
            fluoImage = fluo.clone()
            # get last outputs count
            #if tomo.getObjId() in volIds:
            #    tomogram.count = volIds[tomo.getObjId()]
            #else:
            #    tomogram.count = 0
            fluoImage.count = 0
            fluoList.append(fluoImage)
        print(self.inputFluoImages.get(), fluoList)

        fluoProvider = FluoImagesTreeProvider(fluoList, self.info_path, "json")

        self.dlg = NapariDialog(None, self._getExtraPath(), provider=fluoProvider,)

        # Open dialog to request confirmation to create output
        import tkinter as tk
        frame = tk.Frame()
        if askYesNo(Message.TITLE_SAVE_OUTPUT, Message.LABEL_SAVE_OUTPUT, frame):
            self._createOutput()