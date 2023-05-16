from spfluo.napari.multiple_viewer_widget import annotate_ortho_view

from pyworkflow.gui.dialog import ToolbarListDialog
from pyworkflow.gui.tree import TreeProvider
import pyworkflow.utils as pwutils

import os
import glob
import threading

from spfluo.objects.data import FluoImage


class NapariDialog(ToolbarListDialog):
    """
    taken from scipion-em-emantomo/emantomo/viewers/views_tkinter_tree.py:EmanDialog
    This class extend from ListDialog to allow calling
    a Napari subprocess from a list of FluoImages.
    """

    def __init__(self, parent, path, **kwargs):
        self.path = path
        self.provider = kwargs.get("provider", None)
        ToolbarListDialog.__init__(self, parent,
                                   "Fluoimage List",
                                   allowsEmptySelection=False,
                                   itemDoubleClick=self.doubleClickOnFluoimage,
                                   allowSelect=False,
                                   **kwargs)

    def refresh_gui(self):
        if self.proc.is_alive():
            self.after(1000, self.refresh_gui)
        else:
            # picking is over, counting the number of particles annotated TODO

            # Get file
            outFile = '*%s_info.json' % pwutils.removeBaseExt(self.tomo.getFileName().split("__")[0])
            jsonPath = os.path.join(self.path, "info", outFile)
            jsonPath = glob.glob(jsonPath)[0]

            # count particles and update tree
            jsonDict = loadJson(jsonPath)
            self.fluoimage.count = len(jsonDict["boxes_3d"])
            self.tree.update()

    def doubleClickOnFluoimage(self, e=None):
        self.fluoimage = e
        self.proc = threading.Thread(target=self.lanchNapariForFluoImage, args=(self.fluoimage,))
        self.proc.start()
        self.after(1000, self.refresh_gui)

    def lanchNapariForFluoImage(self, im: FluoImage):
        im_data = im.getData()
        if im_data is None:
            raise Exception(f"{im} is empty! Cannot launch napari.")
        annotate_ortho_view(im_data)

    def _moveCoordsToInfo(self, tomo):
        fnCoor = '*%s_info.json' % pwutils.removeBaseExt(tomo.getFileName().split("__")[0])
        pattern = os.path.join(self.path, fnCoor)
        files = glob.glob(pattern)

        if files:
            infoDir = pwutils.join(os.path.abspath(self.path), 'info')
            pathCoor = os.path.join(infoDir, os.path.basename(files[0]))
            pwutils.makePath(infoDir)
            copyFile(files[0], pathCoor)


class FluoImagesTreeProvider(TreeProvider):
    """ Populate Tree from SetOfFluoImages. """

    def __init__(self, fluoimagesList, path, mode):
        TreeProvider.__init__(self)
        self.fluoimagesList = fluoimagesList
        self._path = path
        self._mode = mode

    def getColumns(self):
        return [('FluoImage', 300), ("# coords", 100), ('status', 150)]

    def getObjectInfo(self, tomo):
        if self._mode == 'txt':
            tomogramName = os.path.basename(tomo.getFileName())
            tomogramName = os.path.splitext(tomogramName)[0]
            filePath = os.path.join(self._path, tomogramName + ".txt")
        elif self._mode == 'json':
            tomogramName = os.path.basename(tomo.getFileName())
            tomogramName = os.path.splitext(tomogramName)[0]

            outFile = '*%s_info.json' % pwutils.removeBaseExt(tomogramName.split("__")[0])
            pattern = os.path.join(self._path, outFile)
            files = glob.glob(pattern)

            filePath = ''
            if files:
                filePath = files[0]

        if not os.path.isfile(filePath):
            return {'key': tomogramName, 'parent': None,
                    'text': tomogramName, 'values': (tomo.count, "TODO"),
                    'tags': ("pending")}
        else:
            return {'key': tomogramName, 'parent': None,
                    'text': tomogramName, 'values': (tomo.count, "DONE"),
                    'tags': ("done")}

    def getObjectPreview(self, obj):
        return (None, None)

    def getObjectActions(self, obj):
        return []

    def _getObjectList(self):
        """Retrieve the object list"""
        return self.fluoimagesList

    def getObjects(self):
        objList = self._getObjectList()
        return objList

    def configureTags(self, tree):
        tree.tag_configure("pending", foreground="red")
        tree.tag_configure("done", foreground="green")