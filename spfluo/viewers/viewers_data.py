from spfluo import objects as fluoobj
from spfluo import Plugin
from spfluo.constants import VISUALISATION_MODULE
from spfluo.convert import save_coordinates3D
from spfluo.objects.data import FluoImage, SetOfCoordinates3D
from spfluo.protocols.protocol_base import ProtFluoBase

from pyworkflow.viewer import Viewer, View, DESKTOP_TKINTER
from pyworkflow.gui.dialog import ToolbarListDialog
from pyworkflow.gui.tree import TreeProvider
from pyworkflow.utils.process import runJob

import os
import threading
from typing import List


class NapariDataViewer(Viewer):
    """ Wrapper to visualize different type of objects
    with Napari.
    """
    _environments = [DESKTOP_TKINTER]
    _targets = [
        fluoobj.SetOfCoordinates3D
    ]

    def __init__(self, **kwargs):
        Viewer.__init__(self, **kwargs)
        self._views = []
    
    def _visualize(self, obj: fluoobj.FluoObject, **kwargs) -> List[View]:
        cls = type(obj)

        if issubclass(cls, fluoobj.SetOfCoordinates3D):
            self._views.append(SetOfCoordinates3DView(self._tkRoot, obj, self.protocol))
        
        return self._views


class CoordinatesTreeProvider(TreeProvider):
    """ Populate Tree from SetOfCoordinates3D. """

    def __init__(self, coords: SetOfCoordinates3D):
        TreeProvider.__init__(self)
        self.coords: SetOfCoordinates3D = coords

    def getColumns(self):
        return [('FluoImage', 300), ("# coords", 100)]

    def getObjectInfo(self, im: FluoImage) -> dict:
        path = im.getFileName()
        im_name, _ = os.path.splitext(os.path.basename(path))
        return {'key': im_name, 'parent': None, 'text': im_name, 'values': im.count}

    def getObjectPreview(self, obj):
        return (None, None)

    def getObjectActions(self, obj):
        return []

    def _getObjectList(self) -> List[FluoImage]:
        """Retrieve the object list"""
        fluoimages = list(self.coords.getPrecedents())
        for im in fluoimages:
            im.count = len(list(self.coords.iterCoordinates(im)))
        return fluoimages

    def getObjects(self):
        objList = self._getObjectList()
        return objList

class SetOfCoordinates3DView(View):
    def __init__(self, parent, coords: SetOfCoordinates3D, protocol: ProtFluoBase):
        self.coords = coords
        self._tkParent = parent
        self._provider = CoordinatesTreeProvider(self.coords)
        self.protocol = protocol
    
    def show(self):
        SetOfCoordinates3DDialog(self._tkParent, self._provider, self.coords, self.protocol)


class SetOfCoordinates3DDialog(ToolbarListDialog):
    """
    taken from scipion-em-emantomo/emantomo/viewers/views_tkinter_tree.py:EmanDialog
    This class extend from ListDialog to allow calling
    a Napari subprocess from a list of FluoImages.
    """

    def __init__(self, parent, provider: CoordinatesTreeProvider, coords: SetOfCoordinates3D, protocol: ProtFluoBase, **kwargs):
        self.provider = provider
        self.coords = coords
        self._protocol = protocol
        ToolbarListDialog.__init__(self, parent,
                                   "Fluoimage List",
                                   self.provider,
                                   allowsEmptySelection=False,
                                   itemDoubleClick=self.doubleClickOnFluoimage,
                                   allowSelect=False,
                                   **kwargs)

    def doubleClickOnFluoimage(self, e=None):
        fluoimage: FluoImage = e
        # Yes, creating a set of coordinates is not easy
        coords_im: SetOfCoordinates3D = self._protocol._createSetOfCoordinates3D(
            self.coords.getPrecedents(),
            self._protocol._getOutputSuffix(SetOfCoordinates3D)
        )
        coords_im.setBoxSize(self.coords.getBoxSize())
        for coord in self.coords.iterCoordinates(fluoimage):
            coords_im.append(coord)
        self.proc = threading.Thread(target=self.lanchNapariForFluoImage, args=(fluoimage, coords_im))
        self.proc.start()

    def lanchNapariForFluoImage(self, im: FluoImage, coords_im: SetOfCoordinates3D):
        path = im.getFileName()
        csv_path = self._protocol._getExtraPath("coords.csv")
        save_coordinates3D(coords_im, csv_path)
        args = " ".join([path, "--coords", csv_path])
        fullProgram = Plugin.getFullProgram(Plugin.getProgram(VISUALISATION_MODULE))
        runJob(None, fullProgram, args, env=Plugin.getEnviron())