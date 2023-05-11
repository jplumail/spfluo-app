def getSPFluoEnvName(version):
    return f'spfluo-{version}'

FLUO_ROOT_VAR = "FLUO_ROOT"
CUDA_LIB_VAR = "CUDA_LIB"

SPFLUO_HOME = "SPFLUO_HOME"
SPFLUO_VERSION = "0.0.1"
SPFLUO_ACTIVATION_CMD = f'conda activate {getSPFluoEnvName(SPFLUO_VERSION)}'

PICKING_MODULE = "picking"
AB_INITIO_MODULE = "ab_initio_reconstruction"
REFINEMENT_MODULE = "refinement"
UTILS_MODULE = "utils"

PICKING_WORKING_DIR = "picking"
CROPPING_SUBDIR = "cropped"

PYTHON_VERSION = "3.8"

GITHUB_TOKEN = "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"

SPFLUO_CUDA_LIB = "SPFLUO_CUDA_LIB"

NO_INDEX = 0