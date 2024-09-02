import os

FLUO_ROOT_VAR = "FLUO_ROOT"
CUDA_LIB_VAR = "CUDA_LIB"

SINGLEPARTICLE_HOME = "SINGLEPARTICLE_HOME"
SPFLUO_VERSION = "0.0.1"

PICKING_MODULE = "picking"
AB_INITIO_MODULE = "ab_initio_reconstruction"
REFINEMENT_MODULE = "refinement"
UTILS_MODULE = "utils"
MANUAL_PICKING_MODULE = "manual_picking"
VISUALISATION_MODULE = "visualisation"
REFINEMENT_MODULE = "refinement"

PICKING_WORKING_DIR = "picking"
CROPPING_SUBDIR = "cropped"

PYTHON_VERSION = "3.10"

SPFLUO_CUDA_LIB = "SINGLEPARTICLE_CUDA_LIB"

target_dir = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "_vendored", "TiPi", "target"
)
TIPI_JAR = os.path.join(target_dir, "TiPi-for-spfluo-1.0.jar")
JAVA_XMX = "16G"

FIJI_HOME = "/opt/fiji-linux64/Fiji.app"

DEFAULT_BATCH = 1
