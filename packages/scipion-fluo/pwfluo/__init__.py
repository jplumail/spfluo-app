"""
This modules contains classes related with Fluo
"""

import os

import pyworkflow as pw
import pyworkflow.plugin as pwplugin
import pyworkflow.utils as pwutils
from pyworkflow.protocol import HostConfig, Protocol
from pyworkflow.viewer import Viewer
from pyworkflow.wizard import Wizard

from pwfluo.objects import FluoObject

# Epoch indicates compatible main Scipion version
# major.minor.micro versioning starting with 1.0.0 in the new epoch
__version__ = "3!0.1dev0"
_logo = "icon.png"


class Domain(pwplugin.Domain):
    _name = __name__
    _objectClass = FluoObject
    _protocolClass = Protocol
    _viewerClass = Viewer
    _wizardClass = Wizard
    _baseClasses = globals()


class Plugin(pwplugin.Plugin):
    pass


class Config:
    # scipion-pyworkflow will generate the path $SCIPION_HOME/software/bindings
    # The path will be created in the working directory if SCIPION_HOME is not set
    SCIPION_FLUO_HOME = os.environ.get(
        "SCIPION_HOME", pwutils.expandPattern("~/.scipion-fluo-home")
    )
    # Allowing the user to set SCIPION_FLUO_USERDATA at installation is issue #8
    SCIPION_FLUO_USERDATA = os.environ.get(
        "SCIPION_FLUO_USERDATA", pwutils.expandPattern("~/ScipionFluoUserData")
    )
    # Location of the contents from scipion-fluo-testdata
    SCIPION_FLUO_TESTDATA = os.environ.get("SCIPION_FLUO_TESTDATA", None)

    SCIPION_FLUO_TEST_OUTPUT = os.environ.get(
        "SCIPION_FLUO_TEST_OUTPUT", os.path.join(SCIPION_FLUO_USERDATA, "Tests")
    )


# ----------- Override some pyworkflow config settings ------------------------

# Create Config.SCIPION_FLUO_HOME if it does not already exist. It is required
# pyworkflow.Config
if not os.path.exists(Config.SCIPION_FLUO_HOME):
    os.mkdir(Config.SCIPION_FLUO_HOME)
# Create Config.SCIPION_FLUO_USERDATA if it does not already exist.
if not os.path.exists(Config.SCIPION_FLUO_USERDATA):
    os.mkdir(Config.SCIPION_FLUO_USERDATA)
# Create default hosts.conf
hostsFile = os.path.join(Config.SCIPION_FLUO_USERDATA, "hosts.conf")
if not os.path.exists(hostsFile):
    HostConfig.writeBasic(hostsFile)


os.environ["SCIPION_VERSION"] = "FLUO - " + __version__
os.environ["SCIPION_HOME"] = pw.Config.SCIPION_HOME = Config.SCIPION_FLUO_HOME
os.environ["SCIPION_USER_DATA"] = pw.Config.SCIPION_USER_DATA = (
    Config.SCIPION_FLUO_USERDATA
)
os.environ["SCIPION_HOSTS"] = pw.Config.SCIPION_HOSTS = hostsFile
os.environ["SCIPION_TESTS_OUTPUT"] = pw.Config.SCIPION_TESTS_OUTPUT = (
    Config.SCIPION_FLUO_TEST_OUTPUT
)

pw.Config.setDomain("pwfluo")


Domain.registerPlugin(__name__)
