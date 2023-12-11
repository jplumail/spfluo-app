import os
import platform
import threading
from typing import List

import pyworkflow.utils as pwutils
from pwfluo import objects as pwfluoobj
from pyworkflow.gui.browser import FileBrowserWindow
from pyworkflow.viewer import DESKTOP_TKINTER, View, Viewer

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

    def runProgram(self, args: list[str], cwd=None):
        pwutils.runJob(None, self.getProgram(), args, env=self.getEnviron(), cwd=cwd)


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
            self._views.append(ImageJView(obj.getFileName(), parent=self.parent))
        elif isinstance(obj, pwfluoobj.SetOfImages):
            self._views.append(
                ImageJView([im.getFileName() for im in obj], parent=self.parent)
            )
        return self._views


###############
## ImageJView ##
###############


class ImageJView(View):
    def __init__(self, files: str | list[str], cwd: str | None = None, parent=None):
        if isinstance(files, str):
            self.files = [files]
        else:
            self.files = files
        self.cwd = cwd
        self.imagej = ImageJ(parent=parent)

    def show(self):
        self.thread = threading.Thread(
            target=self.imagej.runProgram, args=self.files, kwargs={"cwd": self.cwd}
        )
        self.thread.start()
