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
import pyworkflow.utils as pwutils
import pwem

from .constants import *

_logo = "icon.png"
_references = ['Fortun2017']


class Plugin(pwem.Plugin):
    _homeVar = SPFLUO_HOME

    @classmethod
    def _defineVariables(cls):
        cls._defineVar(SPFLUO_CUDA_LIB, pwem.Config.CUDA_LIB)

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
        environ.addLibrary(pwem.Config.CUDA_LIB)

        return environ
    
    @classmethod
    def runSPFluo(cls, protocol, program, args, cwd=None, useCpu=False):
        """ Run IsonNet command from a given protocol. """
        fullProgram = '%s && %s && %s' % (cls.getCondaActivationCmd().replace("&&",""),
                                          SPFLUO_ACTIVATION_CMD,
                                          program)
        protocol.runJob(fullProgram, args, env=cls.getEnviron(), cwd=cwd,
                        numberOfMpi=1)

    @classmethod
    def getProgram(cls, program):
        program = f"python -m spfluo.{program}"
        return program

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
        cuda_version = cls.guessCudaVersion(SPFLUO_CUDA_LIB, default="10.1")
        cupy_version = None
        if cuda_version.major == 10 and cuda_version.minor == 1: # Default version returned by guessCudaVersion
            # Maybe the SPFLUO_CUDA_LIB path doesn't contain the version
            try:
                result = subprocess.check_output("nvcc --version", shell=True, text=True)
                cuda_version_str = result.split('\n')[3].split(', ')[1].split(' ')[1]
                cuda_version = cls.guessCudaVersion(SPFLUO_CUDA_LIB, default=cuda_version_str) # gets the default cuda version
            except subprocess.CalledProcessError:
                print("nvcc not in $PATH. Not installing CuPY. Exiting...")
                sys.exit(1)
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
        if cupy_version is not None:
            installCmd.append(f"pip install --default-timeout=100 {cupy_version}")
        
        # Download and install spfluo
        installCmd.append(f"git clone https://jplumail:{GITHUB_TOKEN}@github.com/dfortun2/SPFluo_stage_reconstruction_symmetryC.git")
        installCmd.append("mv SPFluo_stage_reconstruction_symmetryC spfluo")
        installCmd.append("cd spfluo && pip install .")
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
