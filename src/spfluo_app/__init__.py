import os
import sys


def main():
    os.environ["SCIPION_HOME"] = os.path.expanduser(os.path.join("~","scipion"))
    os.environ["SCIPION_USER_DATA"] = os.path.expanduser(os.path.join("~","ScipionFluoUserData"))

    if not os.path.exists(os.environ["SCIPION_HOME"]):
        os.mkdir(os.environ["SCIPION_HOME"])

    # Necessary when distributing on Mac/Linux
    # https://gregoryszorc.com/docs/python-build-standalone/main/quirks.html#tcl-tk-support-files
    os.environ["TCL_LIBRARY"] = os.path.join(os.path.dirname(sys.executable), "..", "lib", "tcl8.6")

    from scipion.__main__ import main as scipion_main

    scipion_main()