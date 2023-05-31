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
import subprocess
import sys
import re
from packaging.version import parse, InvalidVersion
import pyworkflow.utils as pwutils
from pyworkflow import plugin
import pyworkflow as pw
from pyworkflow.utils import  getSubclasses
from pyworkflow.object import Object
from pyworkflow.viewer import Viewer
from pyworkflow.wizard import Wizard
from pyworkflow.protocol import Protocol

from spfluo.objects import FluoObject
from .constants import CUDA_LIB_VAR, FLUO_ROOT_VAR, GITHUB_TOKEN, PYTHON_VERSION, SPFLUO_ACTIVATION_CMD, SPFLUO_CUDA_LIB, SPFLUO_HOME, SPFLUO_VERSION, getSPFluoEnvName

_logo = "icon.png"
_references = ['Fortun2017']


class Config(pw.Config):
    _get = pw.Config._get
    _join = pw.Config._join

    FLUO_ROOT = _join(_get(FLUO_ROOT_VAR, _join(pw.Config.SCIPION_SOFTWARE, 'fluo')))

    # CUDA
    CUDA_LIB = _get(CUDA_LIB_VAR, '/usr/local/cuda/lib64')
    CUDA_BIN = _get('CUDA_BIN', '/usr/local/cuda/bin')


class Domain(plugin.Domain):
    _name = __name__
    _objectClass = FluoObject
    _protocolClass = Protocol
    _viewerClass = Viewer
    _wizardClass = Wizard
    _baseClasses = getSubclasses(FluoObject, globals())


class Plugin(plugin.Plugin):
    _homeVar = SPFLUO_HOME

    @classmethod
    def _defineVariables(cls):
        cls._defineVar(SPFLUO_CUDA_LIB, Config.CUDA_LIB)

    @classmethod
    def getDependencies(cls):
        """ Return a list of dependencies. Include conda if
            activation command was not found. """
        condaActivationCmd = cls.getCondaActivationCmd()
        neededProgs = ['git']
        if not condaActivationCmd:
            neededProgs.append('conda')

        return neededProgs

    @classmethod
    def getEnviron(cls):
        """ Setup the environment variables needed to launch Relion. """
        environ = pwutils.Environ(os.environ)

        # Take Scipion CUDA library path
        environ.addLibrary(Config.CUDA_LIB)

        return environ

    @classmethod
    def getFullProgram(cls, program):
        return f"{cls.getCondaActivationCmd().replace('&&','')} && {SPFLUO_ACTIVATION_CMD} && {program}"
    
    @classmethod
    def runSPFluo(cls, protocol: Protocol, program, args, cwd=None, useCpu=False):
        """ Run SPFluo command from a given protocol. """
        protocol.runJob(cls.getFullProgram(program), args, env=cls.getEnviron(), cwd=cwd, numberOfMpi=1)

    @classmethod
    def getProgram(cls, program):
        program = f"python -m spfluo.{program}"
        return program
    
    @classmethod
    def getNapariProgram(cls):
        return f"python -m napari"

    @classmethod
    def addSPFluoPackage(cls, env):
        SPFLUO_INSTALLED = f"spfluo_{SPFLUO_VERSION}_installed"
        # Create environment for spfluo
        ENV_NAME = getSPFluoEnvName(SPFLUO_VERSION)
        # Install major dependencies
        installCmd = [
            cls.getCondaActivationCmd().replace("&&",""),
            f'conda create -y -n {ENV_NAME} python={PYTHON_VERSION}',
            SPFLUO_ACTIVATION_CMD
        ]

        # Install Cupy (the right version)
        spfluo_cuda_lib: str = cls.getVar(SPFLUO_CUDA_LIB)
        match = re.search('(cuda)-([0-9]+.[0-9])', os.path.realpath(spfluo_cuda_lib))
        if match and len(match.groups()) == 2:
            cuda_version = parse(match.group(2))
        else:
            print("Couldn't find CUDA version from path", os.path.realpath(spfluo_cuda_lib))
            cuda_version = None
        if cuda_version is None:
            # Maybe the SPFLUO_CUDA_LIB path doesn't contain the version
            try:
                result = subprocess.check_output("nvcc --version", shell=True, text=True)
                cuda_version_str = result.split('\n')[3].split(', ')[1].split(' ')[1]
            except subprocess.CalledProcessError:
                print("nvcc not in $PATH. Not installing CuPY. Exiting...")
                sys.exit(1)
            try:
                cuda_version = parse(cuda_version_str)
            except InvalidVersion:
                cuda_version = None
                print(f"Couldn't version cuda version: {cuda_version_str}")
                sys.exit(1)
        
        cupy_version = None
        if cuda_version.major == 10 and cuda_version.minor == 2:
            cupy_version = 'cupy-cuda102'
        elif cuda_version.major == 11:
            if cuda_version.minor == 0 or cuda_version.minor == 1:
                cupy_version = f'cupy-cuda11{cuda_version.minor}'
            else:
                cupy_version = f'cupy-cuda11x'
        elif cuda_version.major == 12:
            cupy_version = f'cupy-cuda12x'
        else:
            print(f"Your CUDA version {cuda_version} doesn't match one of cupy. You need to have versions CUDA 10.2, 11.x or 12.x.")
            sys.exit(1)
        if cupy_version is not None:
            installCmd.append(f"pip install --default-timeout=100 {cupy_version}")
        
        # Download and install spfluo
        installCmd.append(f"git clone https://jplumail:{GITHUB_TOKEN}@github.com/jplumail/SPFluo_stage_reconstruction_symmetryC.git")
        installCmd.append("mv SPFluo_stage_reconstruction_symmetryC spfluo")
        installCmd.append("cd spfluo && pip install .")

        # Temporary solution until this https://github.com/AllenCellModeling/aicsimageio/issues/495 is fixed
        installCmd.append("pip install \"tifffile>=2023.3.15\"")

        # from https://github.com/AllenCellModeling/napari-aicsimageio/tree/main#use-napari-aicsimageio-as-the-reader-for-all-file-formats
        settings_napari = "from napari.settings import get_settings;" \
            "get_settings().plugins.extension2reader = {'*': 'napari-aicsimageio', **get_settings().plugins.extension2reader}"
        installCmd.append(f"python -c \"{settings_napari}\"")

        installCmd.append(f'touch ../{SPFLUO_INSTALLED}')

        pyem_commands = [(" && ".join(installCmd),[SPFLUO_INSTALLED])]

        envPath = os.environ.get('PATH', "")
        installEnvVars = {'PATH': envPath} if envPath else None
        env.addPackage('spfluo', version=SPFLUO_VERSION,
                       tar='void.tgz',
                       commands=pyem_commands,
                       neededProgs=cls.getDependencies(),
                       default=True,
                       vars=installEnvVars)

    @classmethod
    def defineBinaries(cls, env):
        cls.addSPFluoPackage(env)