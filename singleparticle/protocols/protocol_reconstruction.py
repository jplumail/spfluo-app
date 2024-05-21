import os

import pyworkflow.protocol.params as params
from pwfluo.objects import (
    AverageParticle,
    PSFModel,
    SetOfParticles,
)
from pwfluo.protocols import ProtFluoBase
from pyworkflow import BETA
from pyworkflow.protocol import Protocol
from spfluo.utils.reconstruction import main as reconstruction

from singleparticle.convert import read_poses, save_images, save_poses


class ProtSingleParticleReconstruction(Protocol, ProtFluoBase):
    """
    Reconstruction
    """

    _label = "reconstruction"
    _devStatus = BETA
    OUTPUT_NAME = "reconstruction"
    _possibleOutputs = {OUTPUT_NAME: AverageParticle}

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form: params.Form):
        form.addSection("Data params")
        form.addParam(
            "inputParticles",
            params.PointerParam,
            pointerClass="SetOfParticles",
            label="Particles",
            important=True,
            help="Select the input particles.",
        )
        form.addParam(
            "inputPSF",
            params.PointerParam,
            pointerClass="PSFModel",
            label="PSF",
            important=True,
            help="Select the PSF.",
        )
        form.addParam(
            "gpu",
            params.BooleanParam,
            default=True,
            label="Use GPU?",
            help="This protocol can use the GPU.",
        )
        form.addParam(
            "minibatch",
            params.IntParam,
            default=0,
            label="Size of a minibatch",
            expertLevel=params.LEVEL_ADVANCED,
            help="The smaller the size, the less memory will be used.\n"
            "0 for automatic minibatch.",
            condition="gpu",
        )
        form.addParam(
            "pad",
            params.BooleanParam,
            default=True,
            expertLevel=params.LEVEL_ADVANCED,
            label="Pad particles?",
        )
        form.addSection(label="Reconstruction params")
        form.addParam(
            "sym",
            params.IntParam,
            default=1,
            label="Symmetry degree",
            help="Adds a cylindrical symmetry constraint.",
        )
        form.addParam(
            "lbda",
            params.FloatParam,
            default=100.0,
            label="Lambda",
            help="Higher results in smoother results.",
        )

    def _insertAllSteps(self):
        self.root_dir = os.path.abspath(self._getExtraPath("root"))
        self.outputDir = os.path.abspath(self._getExtraPath("working_dir"))
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.outputDir, exist_ok=True)
        self.psfPath = os.path.join(self.root_dir, "psf.ome.tiff")
        self.final_reconstruction = os.path.abspath(
            self._getExtraPath("final_reconstruction.ome.tiff")
        )
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.reconstructionStep)
        self._insertFunctionStep(self.createOutputStep)

    def prepareStep(self):
        # Image links for particles
        particles: SetOfParticles = self.inputParticles.get()
        psf: PSFModel = self.inputPSF.get()
        save_poses(os.path.join(self.root_dir, "poses.csv"), particles)

        max_dim = particles.getMaxDataSize()
        if self.pad:
            max_dim = max_dim * (2**0.5)

        # Make isotropic
        vs = particles.getVoxelSize()
        pixel_size = min(vs)

        images_paths = save_images(
            [psf] + list(particles.iterItems()),
            self.root_dir,
            (max_dim, max_dim, max_dim),
            voxel_size=(pixel_size, pixel_size),
        )

        os.link(images_paths[0], os.path.join(self.root_dir, "psf.ome.tiff"))

    def reconstructionStep(self):
        poses_path = os.path.join(self.root_dir, "poses.csv")
        particles_paths = [
            os.path.join(self.root_dir, objId + ".ome.tiff")
            for _, objId in read_poses(poses_path)
        ]

        minibatch = self.minibatch.get() if self.minibatch.get() > 0 else None
        reconstruction(
            particles_paths,
            poses_path,
            self.psfPath,
            self.final_reconstruction,
            self.lbda.get(),
            self.sym.get(),
            self.gpu.get(),
            minibatch,
        )

    def createOutputStep(self):
        vs = min(self.inputParticles.get().getVoxelSize())
        reconstruction = AverageParticle.from_filename(
            self.final_reconstruction, voxel_size=(vs, vs)
        )

        self._defineOutputs(**{self.OUTPUT_NAME: reconstruction})
