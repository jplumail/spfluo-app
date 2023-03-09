def getSPFluoEnvName(version):
    return f'spfluo-{version}'


SPFLUO_HOME = "SPFLUO_HOME"
SPFLUO_VERSION = "0.0.1"
SPFLUO_ACTIVATION_CMD = f'conda activate {getSPFluoEnvName(SPFLUO_VERSION)}'

PICKING_MODULE = "picking"
AB_INITIO_MODULE = "ab_initio_reconstruction"
REFINEMENT_MODULE = "refinement"

PICKING_WORKING_DIR = "picking"
CROPPING_SUBDIR = "cropped"

PYTHON_VERSION = "3.8"

GITHUB_TOKEN = "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"

SPFLUO_CUDA_LIB = "SPFLUO_CUDA_LIB"