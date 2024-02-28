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

from singleparticle import Plugin
from singleparticle.constants import UTILS_MODULE
from singleparticle.convert import read_poses, save_image, save_particles_and_poses


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
        particles_paths, max_dim = save_particles_and_poses(self.root_dir, particles)

        # PSF Path
        psf: PSFModel = self.inputPSF.get()
        save_image(self.psfPath, psf)

        # Make isotropic
        vs = particles.getVoxelSize()
        if vs is None:
            raise RuntimeError("Input Particles don't have a voxel size.")

        input_paths = particles_paths + [self.psfPath]
        args = ["-f", "isotropic_resample"]
        args += ["-i"] + input_paths
        folder_isotropic = os.path.abspath(self._getExtraPath("isotropic"))
        if not os.path.exists(folder_isotropic):
            os.makedirs(folder_isotropic, exist_ok=True)
        args += ["-o", f"{folder_isotropic}"]
        args += ["--spacing", f"{vs[1]}", f"{vs[0]}", f"{vs[0]}"]
        Plugin.runJob(self, Plugin.getSPFluoProgram(UTILS_MODULE), args=args)

        # Pad
        input_paths = [
            os.path.join(folder_isotropic, f) for f in os.listdir(folder_isotropic)
        ]
        if self.pad:
            max_dim = int(max_dim * (2**0.5)) + 1
        folder_resized = os.path.abspath(self._getExtraPath("isotropic_cropped"))
        if not os.path.exists(folder_resized):
            os.makedirs(folder_resized, exist_ok=True)
        args = ["-f", "resize"]
        args += ["-i"] + input_paths
        args += ["--size", f"{max_dim}"]
        args += ["-o", f"{folder_resized}"]
        Plugin.runJob(self, Plugin.getSPFluoProgram(UTILS_MODULE), args=args)

        # Links
        os.remove(self.psfPath)
        for p in particles_paths:
            os.remove(p)
        # Link to psf
        os.link(
            os.path.join(folder_resized, os.path.basename(self.psfPath)), self.psfPath
        )
        # Links to particles
        for p in particles_paths:
            os.link(os.path.join(folder_resized, os.path.basename(p)), p)

    def reconstructionStep(self):
        poses_path = os.path.join(self.root_dir, "poses.csv")
        particles_paths = [
            os.path.join(self.root_dir, fname) for _, fname in read_poses(poses_path)
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
