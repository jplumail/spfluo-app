from spfluo import objects as fluoobj
from spfluo import Plugin
from spfluo.constants import VISUALISATION_MODULE
from spfluo.convert import save_coordinates3D
from spfluo.objects.data import FluoImage, SetOfCoordinates3D
from spfluo.viewers.view_picking import FluoImagesTreeProvider

from pyworkflow.viewer import Viewer, View, DESKTOP_TKINTER
from pyworkflow.gui.dialog import ToolbarListDialog
from pyworkflow.utils.process import runJob
from pyworkflow.protocol import Protocol

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


class SetOfCoordinates3DView(View):
    def __init__(self, parent, coords: SetOfCoordinates3D, protocol: Protocol):
        self.coords = coords
        self._tkParent = parent
        self._provider = FluoImagesTreeProvider(list(self.coords.getPrecedents()))
        self.protocol = protocol
    
    def show(self):
        SetOfCoordinates3DDialog(self._tkParent, self._provider, self.coords, self.protocol)


class SetOfCoordinates3DDialog(ToolbarListDialog):
    """
    taken from scipion-em-emantomo/emantomo/viewers/views_tkinter_tree.py:EmanDialog
    This class extend from ListDialog to allow calling
    a Napari subprocess from a list of FluoImages.
    """

    def __init__(self, parent, provider: FluoImagesTreeProvider, coords: SetOfCoordinates3D, protocol: Protocol, **kwargs):
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
        coords_im = SetOfCoordinates3D()
        print(self.coords)
        print(self.coords.getImgIds(), fluoimage.getImgId())
        for coord in self.coords.iterCoordinates(fluoimage):
            print(coord)
            coords_im.append(coord)
        self.proc = threading.Thread(target=self.lanchNapariForFluoImage, args=(fluoimage, coords_im))

    def lanchNapariForFluoImage(self, im: FluoImage, coords_im: SetOfCoordinates3D):
        path = im.getFileName()
        csv_path = self._protocol._getExtraPath("coords.csv")
        save_coordinates3D(coords_im, csv_path)
        args = " ".join([path, "--coords", csv_path])
        fullProgram = Plugin.getFullProgram(Plugin.getProgram(VISUALISATION_MODULE))
        runJob(None, fullProgram, args, env=Plugin.getEnviron())