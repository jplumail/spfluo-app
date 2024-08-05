from __future__ import annotations

import logging
import os
import platform
import tempfile
import threading
from typing import List

import pyworkflow.utils as pwutils
import tifffile
from pwfluo import objects as pwfluoobj
from pwfluo.constants import MICRON_STR
from pyworkflow.gui.browser import FileBrowserWindow
from pyworkflow.viewer import DESKTOP_TKINTER, View, Viewer

from singleparticle.constants import FIJI_HOME

logger = logging.getLogger("imagej-viewer")

handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


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
            for i, im in enumerate(images):
                temp_file = os.path.join(temp_dir, f"temp-{i}.ome.tiff")
                with tifffile.TiffWriter(temp_file, ome=True, bigtiff=True) as tif:
                    multichannel = im.getNumChannels() > 1
                    vs_xy, vs_z = im.getVoxelSize() if im.getVoxelSize() else (1, 1)
                    metadata = {
                        "axes": "CZYX",
                        "PositionT": i,
                        "PhysicalSizeX": vs_xy,
                        "PhysicalSizeXUnit": MICRON_STR,
                        "PhysicalSizeY": vs_xy,
                        "PhysicalSizeYUnit": MICRON_STR,
                        "PhysicalSizeZ": vs_z,
                        "PhysicalSizeZUnit": MICRON_STR,
                    }
                    logger.debug(f"{im.getVoxelSize()=}, {type(im.getVoxelSize())=}")
                    logger.debug(f"voxel size: {vs_xy}x{vs_xy}x{vs_z}")
                    data = im.getData()
                    X, Y, Z = im.getDim()
                    if data is not None:
                        tif.write(data, metadata=metadata, contiguous=False)
                    else:
                        raise ValueError(f"Data is None for {im}.")
            first_temp_file = os.path.join(temp_dir, "temp-0.ome.tiff")
            temp_files_pattern = (
                os.path.join(temp_dir, f"temp-<0-{len(images)}>.ome.tiff")
                if len(images) > 1
                else os.path.join(temp_dir, "temp-0.ome.tiff")
            )
            color_mode = "Composite" if multichannel else "Default"
            if os.name == "nt":
                first_temp_file = os.fspath(first_temp_file).replace("\\", "\\\\")
                temp_files_pattern = os.fspath(temp_files_pattern).replace("\\", "\\\\")
            swap_dims_options = (
                (
                    f"dimensions axis_1_number_of_images={len(images)} "
                    "axis_1_axis_first_image=0 axis_1_axis_increment=1 "
                )
                if len(images) > 1
                else ""
            )
            script = (
                "run('Bio-Formats', "
                f"'open={first_temp_file} "
                f"autoscale color_mode={color_mode} group_files "
                "rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT "
                f"swap_dimensions "
                f"{swap_dims_options}"
                f"contains=[] name={temp_files_pattern} "
                f"z_1={Z} c_1=1 t_1={len(images)}');"
            )
            logger.debug(script)
            pwutils.runJob(
                None,
                self.getProgram(),
                ["-eval", script],
                env=self.getEnviron(),
                cwd=cwd,
            )


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
            self._views.append(ImageJView([im for im in obj], parent=self.parent))
        return self._views


###############
## ImageJView ##
###############


class ImageJView(View):
    def __init__(
        self,
        images: pwfluoobj.Image | list[pwfluoobj.Image],
        cwd: str | None = None,
        parent=None,
    ):
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
