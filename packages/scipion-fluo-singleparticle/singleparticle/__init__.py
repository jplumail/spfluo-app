# **************************************************************************
# *
# * Authors:     Jean Plumail (jplumail@unistra.fr)
# *
# * ICube, University of Strasburg
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

import os
import sys
import threading
from typing import List, Union

import pyworkflow as pw
import pyworkflow.utils as pwutils
import spfluo
from pwfluo.objects import FluoObject
from pyworkflow import plugin
from pyworkflow.protocol import Protocol
from pyworkflow.utils import getSubclasses
from pyworkflow.viewer import Viewer
from pyworkflow.wizard import Wizard
from waitress import serve

from singleparticle.constants import (
    FLUO_ROOT_VAR,
    JAVA_XMX,
    SINGLEPARTICLE_HOME,
    TIPI_JAR,
)
from singleparticle.web.server import app

_logo = "icon.png"
_references = ["Fortun2017"]


class Config(pw.Config):
    _get = pw.Config._get
    _join = pw.Config._join

    FLUO_ROOT = _join(_get(FLUO_ROOT_VAR, _join(pw.Config.SCIPION_SOFTWARE, "fluo")))


class Domain(plugin.Domain):
    _name = __name__
    _objectClass = FluoObject
    _protocolClass = Protocol
    _viewerClass = Viewer
    _wizardClass = Wizard
    _baseClasses = getSubclasses(FluoObject, globals())


class Plugin(plugin.Plugin):
    _homeVar = SINGLEPARTICLE_HOME

    @classmethod
    def getDependencies(cls):
        """Return a list of dependencies. Include conda if
        activation command was not found."""
        return []

    @classmethod
    def getEnviron(cls):
        """Setup the environment variables needed to launch Relion."""
        environ = pwutils.Environ(os.environ)

        return environ

    @classmethod
    def runJob(cls, protocol: Protocol, program, args, cwd=None, useCpu=False):
        """Run SPFluo command from a given protocol."""
        protocol.runJob(
            program,
            args,
            env=cls.getEnviron(),
            cwd=cwd,
            numberOfMpi=1,
        )

    @classmethod
    def getPythonPath(cls):
        return sys.executable

    @classmethod
    def getSPFluoProgram(cls, program: Union[str, List[str]]):
        if isinstance(program, str):
            program = [program]
        command = f"{cls.getPythonPath()} -m spfluo"
        for p in program:
            command += f".{p}"
        return command

    @classmethod
    def getNapariProgram(cls):
        return f"{cls.getPythonPath()} -m napari"

    @classmethod
    def getMicroTipiProgram(cls, program: str):
        microtipi_cmd = f"java -Xmx{JAVA_XMX} -jar {TIPI_JAR}"
        programs = ["deconv", "blinddeconv"]
        assert program in programs, f"{program} should be one of {programs}."
        microtipi_cmd += " " + program
        return microtipi_cmd

    @classmethod
    def addSingleParticlePackage(cls, env):
        env.addPackage(
            "singleparticle",
            version=spfluo.__version__,
            tar="void.tgz",
            commands=[],
            neededProgs=cls.getDependencies(),
            default=True,
            vars=None,
        )

    @classmethod
    def defineBinaries(cls, env):
        cls.addSingleParticlePackage(env)


def run_server():
    try:
        serve(app, host="127.0.0.1", port=5000)
    except OSError:
        pass


# Start Waitress server in a separate thread
threading.Thread(target=run_server, daemon=True).start()
