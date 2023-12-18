import os
import platform
import tempfile
import threading
from typing import List

import pyworkflow.utils as pwutils
from pwfluo import objects as pwfluoobj
from pyworkflow.gui.browser import FileBrowserWindow
from pyworkflow.viewer import DESKTOP_TKINTER, View, Viewer
import tifffile

from singleparticle.constants import FIJI_HOME


class ImageJ:
    def __init__(self, parent=None):
        self._home = FIJI_HOME
        self.parent = parent
        self.getHome()

    def ask_home(self) -> None:
        def onSelect(obj: str):
            print(obj)
            self._home = obj.getPath()
            print(self._home)

        browser = FileBrowserWindow(
            "Fiji Home", self.parent, "/home/plumail", onSelect=onSelect
        )
        browser.show()

    def getHome(self):
        return self._home

    def getEnviron(self):
        environ = pwutils.Environ(os.environ)
        environ.set("PATH", self._home, position=pwutils.Environ.BEGIN)
        return environ

    def getProgram(self):
        return (
            f"ImageJ-"
            f"{'linux' if platform.system()=='Linux' else 'win'}"
            f"{'64' if platform.architecture()[0]=='64bit' else '32'}"
            f"{'' if platform.system()=='Linux' else '.exe'}"
        )
    
    def runProgram(self, images: list[pwfluoobj.Image], cwd=None):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "temp.ome.tiff")
            with tifffile.TiffWriter(temp_file, ome=True, bigtiff=True) as tif:
                for i, im in enumerate(images):
                    vs_xy, vs_z = im.getVoxelSize() if im.getVoxelSize() else (1, 1)
                    metadata = {
                        'axes': 'CZYX',
                        'PositionT': i,
                        'PhysicalSizeX': vs_xy,
                        'PhysicalSizeXUnit': 'µm',
                        'PhysicalSizeY': vs_xy,
                        'PhysicalSizeYUnit': 'µm',
                        'PhysicalSizeZ': vs_z,
                        'PhysicalSizeZUnit': 'µm',
                    }
                    data = im.getData()
                    if data is not None:
                        tif.write(data, metadata=metadata, contiguous=False)
                    else:
                        raise ValueError(f"Data is None for {im}.")
            series_str = " ".join(["series_"+str(i+1) for i in range(len(images))])
            script = ("run('Bio-Formats', "
            f"'open={temp_file} autoscale color_mode=Default "
            "concatenate_series rois_import=[ROI manager] view=Hyperstack "
            f"stack_order=XYCZT {series_str}');")
            pwutils.runJob(None, self.getProgram(), ["-eval", script],
                           env=self.getEnviron(), cwd=cwd)



class ImageJViewer(Viewer):
    """Wrapper to visualize different type of objects
    with ImageJ.
    """

    _environments = [DESKTOP_TKINTER]
    _targets = [
        pwfluoobj.Image,
        pwfluoobj.SetOfImages,
    ]

    def __init__(self, **kwargs):
        Viewer.__init__(self, **kwargs)
        self._views: List[View] = []
        self.parent = kwargs.get("parent", None)

    def _visualize(self, obj: pwfluoobj.FluoObject, **kwargs):
        if isinstance(obj, pwfluoobj.Image):
            self._views.append(ImageJView(obj, parent=self.parent))
        elif isinstance(obj, pwfluoobj.SetOfImages):
            self._views.append(
                ImageJView([im for im in obj], parent=self.parent)
            )
        return self._views


###############
## ImageJView ##
###############


class ImageJView(View):
    def __init__(
            self,
            images: pwfluoobj.Image | list[pwfluoobj.Image],
            cwd: str | None = None, parent=None):
        if isinstance(images, pwfluoobj.Image):
            self.images = [images]
        else:
            self.images = images
        self.cwd = cwd
        self.imagej = ImageJ(parent=parent)


    def show(self):
        self.thread = threading.Thread(
            target=self.imagej.runProgram, args=[self.images], kwargs={"cwd": self.cwd}
        )
        self.thread.start()
