import os
from enum import Enum

import pyworkflow.protocol.params as params
from pwfluo.objects import AverageParticle, PSFModel, SetOfCoordinates3D, SetOfParticles
from pwfluo.protocols import ProtFluoBase
from pyworkflow import BETA
from pyworkflow.protocol import Protocol

from singleparticle import Plugin
from singleparticle.constants import REFINEMENT_MODULE, UTILS_MODULE
from singleparticle.convert import save_particles_and_poses, save_psf


class outputs(Enum):
    reconstructedVolume = AverageParticle
    coordinates = SetOfCoordinates3D


class ProtSingleParticleRefinement(Protocol, ProtFluoBase):
    """
    Refinement
    """

    _label = "refinement"
    _devStatus = BETA
    _possibleOutputs = outputs

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
            "channel",
            params.IntParam,
            default=0,
            label="Reconstruct on channel?",
            help="This protocol reconstruct an average particle in one channel only.",
        )
        form.addParam(
            "gpu",
            params.BooleanParam,
            default=False,
            expertLevel=params.LEVEL_ADVANCED,
            label="Use GPU?",
            help="This protocol can use the GPU but it's unstable.",
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
        form.addParam(
            "ranges",
            params.StringParam,
            label="Ranges",
            help="Sequence of angle ranges, in degrees.",
            default="40 20 10 5",
        )
        form.addParam(
            "steps",
            params.StringParam,
            label="Steps",
            help="Number of steps in the range to create the discretization",
            default="10 10 10 10",
        )
        form.addParam(
            "N_axes",
            params.IntParam,
            default=25,
            label="N axes",
            expertLevel=params.LEVEL_ADVANCED,
            help="For the first iteration, number of axes for the discretization of the"
            "sphere.",
        )
        form.addParam(
            "N_rot",
            params.IntParam,
            default=20,
            label="N rot",
            expertLevel=params.LEVEL_ADVANCED,
            help="For the first iteration, number of rotation per axis for the"
            "discretization of the sphere.",
        )

    def _insertAllSteps(self):
        self.root_dir = os.path.abspath(self._getExtraPath("root"))
        self.outputDir = os.path.abspath(self._getExtraPath("working_dir"))
        self.psfPath = os.path.join(self.root_dir, "psf.tif")
        self.final_reconstruction = os.path.abspath(
            self._getExtraPath("final_reconstruction.tif")
        )
        self.final_poses = os.path.abspath(self._getExtraPath("final_poses.csv"))
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.reconstructionStep)
        self._insertFunctionStep(self.createOutputStep)

    def prepareStep(self):
        # Image links for particles
        particles: SetOfParticles = self.inputParticles.get()
        channel = self.channel.get() if particles.getNumChannels() > 0 else None
        particles_paths, max_dim = save_particles_and_poses(
            self.root_dir, particles, channel=channel
        )

        # PSF Path
        psf: PSFModel = self.inputPSF.get()
        save_psf(self.psfPath, psf)

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
        Plugin.runSPFluo(self, Plugin.getProgram(UTILS_MODULE), args=args)

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
        Plugin.runSPFluo(self, Plugin.getProgram(UTILS_MODULE), args=args)

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
        ranges = "0 " + str(self.ranges) if len(str(self.ranges)) > 0 else "0"
        args = [
            "--particles_dir",
            os.path.join(self.root_dir, "particles"),
            "--psf_path",
            self.psfPath,
            "--guessed_poses_path",
            os.path.join(self.root_dir, "poses.csv"),
            "--ranges",
            *ranges.split(" "),
            "--steps",
            f"({self.N_axes},{self.N_rot})",
        ]
        if len(str(self.steps)) > 0:
            args += str(self.steps).split(" ")
        args += [
            "--output_reconstruction_path",
            self.final_reconstruction,
            "--output_poses_path",
            self.final_poses,
            "-l",
            self.lbda.get(),
            "--symmetry",
            self.sym.get(),
        ]
        if self.gpu:
            args += ["--gpu"]
        Plugin.runSPFluo(self, Plugin.getProgram(REFINEMENT_MODULE), args=args)

    def createOutputStep(self):
        # Output 1 : reconstruction Volume
        reconstruction = AverageParticle()
        reconstruction.setFileName(self.final_reconstruction)
        self._defineOutputs(**{outputs.reconstructedVolume.name: reconstruction})
