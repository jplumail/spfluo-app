import os
from enum import Enum

import pyworkflow.protocol.params as params
from pwfluo.objects import (
    AverageParticle,
    Particle,
    PSFModel,
    SetOfParticles,
)
from pwfluo.protocols import ProtFluoBase
from pyworkflow import BETA
from pyworkflow.protocol import Protocol

from singleparticle import Plugin
from singleparticle.constants import DEFAULT_BATCH, REFINEMENT_MODULE


class outputs(Enum):
    reconstructedVolume = AverageParticle
    particles = SetOfParticles


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
            "downsampling_factor",
            params.FloatParam,
            default=1,
            label="downsampling factor",
            help="Downsample all images by a certain factor. Can speed up the protocol."
            " A downsampling factor of 2 will divide the size of the images by 2 "
            "(and speed up computation by ~8!).",
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
            "initialVolume",
            params.PointerParam,
            pointerClass="Particle",
            label="Initial volume",
            expertLevel=params.LEVEL_ADVANCED,
            allowsNull=True,
        )
        form.addParam(
            "gpu",
            params.BooleanParam,
            default=True,
            label="Use GPU?",
            help="This protocol can use the GPU but it's unstable.",
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
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.outputDir, exist_ok=True)
        self.psfPath = os.path.join(self.root_dir, "psf.ome.tiff")
        self.initial_volume_path = os.path.join(
            self.root_dir, "initial_volume.ome.tiff"
        )
        self.final_reconstruction = os.path.abspath(
            self._getExtraPath("final_reconstruction.ome.tiff")
        )
        self.final_poses = os.path.abspath(self._getExtraPath("final_poses.csv"))
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.reconstructionStep)
        self._insertFunctionStep(self.createOutputStep)

    def prepareStep(self):
        # Image links for particles
        particles: SetOfParticles = self.inputParticles.get()
        psf: PSFModel = self.inputPSF.get()
        save_poses(os.path.join(self.root_dir, "poses.csv"), particles)

        images = [psf] + list(particles.iterItems())
        initial_volume: Particle | None = self.initialVolume.get()
        if initial_volume:
            images += [initial_volume]

        max_dim = particles.getMaxDataSize()
        if self.pad:
            max_dim = max_dim * (2**0.5)

        # Make isotropic
        vs = particles.getVoxelSize()
        pixel_size = min(vs)
        # Downsample
        pixel_size = pixel_size * self.downsampling_factor.get()

        images_paths = save_images(
            images,
            self.root_dir,
            (max_dim, max_dim, max_dim),
            voxel_size=(pixel_size, pixel_size),
        )

        if initial_volume:
            os.link(images_paths[-1], self.initial_volume_path)
        os.link(images_paths[0], self.psfPath)

        os.makedirs(os.path.join(self.root_dir, "particles"), exist_ok=True)
        for _, objId in read_poses(os.path.join(self.root_dir, "poses.csv")):
            os.link(
                os.path.join(self.root_dir, objId + ".ome.tiff"),
                os.path.join(self.root_dir, "particles", objId + ".ome.tiff"),
            )

    def reconstructionStep(self):
        ranges = "0 " + str(self.ranges) if len(str(self.ranges)) > 0 else "0"
        args = []
        if self.initialVolume.get():
            args += ["--initial_volume_path", self.initial_volume_path]
        args += [
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
            if self.minibatch.get() > 0:
                args += ["--minibatch", self.minibatch.get()]
        Plugin.runJob(self, Plugin.getSPFluoProgram(REFINEMENT_MODULE), args=args)

    def createOutputStep(self):
        # Output 1 : reconstruction Volume
        vs = min(self.inputParticles.get().getVoxelSize())
        reconstruction = AverageParticle.from_filename(
            self.final_reconstruction, voxel_size=(vs, vs)
        )

        # Output 2 : particles rotated
        output_particles = self._createSetOfParticles()
        transforms = {img_id: t for t, img_id in read_poses(self.final_poses)}
        for particle in self.inputParticles.get():
            particle: Particle

            rotated_transform = transforms[str(particle.getObjId())]  # new coords

            # New file (link to particle)
            rotated_particle_path = self._getExtraPath(particle.getBaseName())
            os.link(particle.getFileName(), rotated_particle_path)

            # Creating the particle
            rotated_particle = Particle.from_filename(
                rotated_particle_path, voxel_size=particle.getVoxelSize()
            )
            rotated_particle.setTransform(rotated_transform)
            rotated_particle.setImageName(particle.getImageName())
            rotated_particle.setImgId(os.path.basename(rotated_particle_path))

            output_particles.append(rotated_particle)

        output_particles.write()

        self._defineOutputs(
            **{
                outputs.reconstructedVolume.name: reconstruction,
                outputs.particles.name: output_particles,
            }
        )
