from __future__ import annotations

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

from singleparticle.constants import DEFAULT_BATCH
from singleparticle.convert import (
    save_image,
    save_particles_and_poses,
)


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
            default=DEFAULT_BATCH,
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
            default=1.0,
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

        max_dim = particles.getMaxDataSize()
        if self.pad:
            max_dim = max_dim * (2**0.5)

        # Make isotropic
        vs = particles.getVoxelSize()
        pixel_size = min(vs)

        common_size = (max_dim, max_dim, max_dim)
        self.common_pixel_size = (pixel_size, pixel_size)
        self.particles_path, _, self.poses_path, _ = save_particles_and_poses(
            self.root_dir,
            particles,
            common_size,
            channel=None,
            voxel_size=self.common_pixel_size,
        )
        save_image(
            self.psfPath,
            psf,
            common_size,
            channel=None,
            voxel_size=self.common_pixel_size,
        )

    def reconstructionStep(self):
        minibatch = self.minibatch.get() if self.minibatch.get() > 0 else None
        reconstruction(
            self.particles_path,
            self.poses_path,
            self.psfPath,
            self.final_reconstruction,
            self.lbda.get(),
            self.sym.get(),
            self.gpu.get(),
            minibatch,
        )

    def createOutputStep(self):
        reconstruction = AverageParticle.from_filename(
            self.final_reconstruction, voxel_size=self.common_pixel_size
        )

        self._defineOutputs(**{self.OUTPUT_NAME: reconstruction})
