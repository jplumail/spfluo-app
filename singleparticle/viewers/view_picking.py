import os
import threading
from typing import List, Optional

from pwfluo.objects import FluoImage
from pyworkflow.gui.dialog import ToolbarListDialog
from pyworkflow.gui.tree import TreeProvider
from pyworkflow.protocol import Protocol
from pyworkflow.viewer import View


class PickingTreeProvider(TreeProvider):
    """Populate Tree from SetOfFluoImages."""

    def __init__(self, fluoimagesList: List[FluoImage]):
        TreeProvider.__init__(self)
        self.fluoimagesList: List[FluoImage] = fluoimagesList

    def getColumns(self):
        return [("FluoImage", 300), ("# coords", 100), ("status", 150)]

    def getObjectInfo(self, im: FluoImage) -> Optional[dict]:
        path = im.getFileName()
        im_name, _ = os.path.splitext(os.path.basename(path))
        d = {"key": im_name, "parent": None, "text": im_name}
        if im.in_viewer:
            status_text = "IN PROGRESS"
            d["tags"] = "in progress"
        elif im.count > 0:
            status_text = "DONE"
            d["tags"] = "done"
        else:
            status_text = "TODO"
            d["tags"] = "pending"
        d["values"] = (im.count, status_text)
        return d

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
        tree.tag_configure("in progress", foreground="black")


class PickingDialog(ToolbarListDialog):
    """
    taken from scipion-em-emantomo/emantomo/viewers/views_tkinter_tree.py:EmanDialog
    This class extend from ListDialog to allow calling
    a Napari subprocess from a list of FluoImages.
    """

    def __init__(
        self, parent, provider: PickingTreeProvider, protocol: Protocol, **kwargs
    ):
        self.size = 10
        self.spacing = None
        for im in provider.fluoimagesList:
            ps = im.getVoxelSize()
            if ps is not None:
                if self.spacing is None:
                    self.spacing = ps
                else:
                    if self.spacing == ps:
                        continue
                    else:
                        raise ValueError(
                            f"Images from {provider.fluoimagesList} don't have"
                            f"the same pixel size. Got {self.spacing} and {ps}."
                        )
        self._protocol = protocol
        self.provider = provider
        self.refresh_gui(initialized=False)
        ToolbarListDialog.__init__(
            self,
            parent,
            "Fluoimage List",
            self.provider,
            allowsEmptySelection=False,
            itemDoubleClick=self.doubleClickOnFluoimage,
            allowSelect=False,
            **kwargs,
        )

    def refresh_gui(self, initialized=True):
        for im in self.provider.fluoimagesList:
            _, csv_path = self._protocol.getCsvPath(im)
            if os.path.isfile(csv_path):
                # count number of lines in csv file
                with open(csv_path, "r") as f:
                    count = sum(1 for line in f)
                if count > 0:
                    im.count = count - 1
                    with open(csv_path, "r") as f:
                        next(f)  # skip header
                        line = f.readline()
                        line = line.split(",")
                        if len(line) == 5:  # verify if csv has data
                            self.size = int(
                                float(line[4])
                            )  # last column is the boxsize
        if initialized:
            if not self.proc.is_alive():
                self.fluoimage.in_viewer = False
            self.tree.update()
            self.after(1000, self.refresh_gui)

    def doubleClickOnFluoimage(self, e=None):
        self.fluoimage = e
        self.fluoimage.in_viewer = True
        self.proc = threading.Thread(
            target=self.lanchNapariForFluoImage, args=(self.fluoimage,)
        )
        self.proc.start()
        self.after(1000, self.refresh_gui)

    def lanchNapariForFluoImage(self, im: FluoImage):
        from singleparticle import Plugin
        from singleparticle.constants import MANUAL_PICKING_MODULE

        path, csv_path = self._protocol.getCsvPath(im)
        args = [
            path,
            csv_path,
            "--spacing",
            str(self.spacing[1]),
            str(self.spacing[0]),
            str(self.spacing[0]),
        ]
        Plugin.runJob(
            self._protocol, Plugin.getSPFluoProgram(MANUAL_PICKING_MODULE), args
        )


class PickingView(View):
    """This class implements a view using Tkinter ListDialog
    and the PickingTreeProvider.
    """

    def __init__(self, parent, protocol: Protocol, fluoList: List[FluoImage], **kwargs):
        self._tkParent = parent
        self._protocol = protocol
        self._provider = PickingTreeProvider(fluoList)

    def show(self):
        PickingDialog(self._tkParent, self._provider, self._protocol)
