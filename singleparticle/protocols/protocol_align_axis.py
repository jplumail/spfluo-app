import os
from enum import Enum

from pwfluo.objects import (
    Particle,
    SetOfParticles,
)
from pwfluo.protocols import ProtFluoBase
from pyworkflow import BETA
from pyworkflow.object import RELATION_TRANSFORM
from pyworkflow.protocol import Form, Protocol, params

from singleparticle import Plugin
from singleparticle.constants import UTILS_MODULE
from singleparticle.convert import read_poses, save_poses


class outputs(Enum):
    aligned_volume = Particle
    aligned_particles = SetOfParticles


class ProtSingleParticleAlignAxis(Protocol, ProtFluoBase):
    """Align the axis of a cylindrical particle"""

    OUTPUT_NAME = "aligned SetOfParticles"
    _label = "align axis"
    _devStatus = BETA
    _possibleOutputs = outputs

    def _defineParams(self, form: Form):
        form.addSection(label="Input")
        form.addParam(
            "inputParticle",
            params.PointerParam,
            pointerClass="Particle",
            label="Particle to align",
            important=True,
        )
        form.addParam(
            "inputParticles",
            params.PointerParam,
            pointerClass="SetOfParticles",
            label="particles to rotate",
        )

    def _insertAllSteps(self):
        self.poses_csv = self._getExtraPath("poses.csv")
        self.rotated_poses_csv = self._getExtraPath("new_poses.csv")
        self.rotated_particle_path = self._getExtraPath("rotated-volume.tiff")
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.alignStep)
        self._insertFunctionStep(self.createOuputStep)

    def prepareStep(self):
        self.input_particles: SetOfParticles = self.inputParticles.get()
        save_poses(self.poses_csv, self.input_particles)

    def alignStep(self):
        self.particle: Particle = self.inputParticle.get()
        particle_path = self.particle.getFileName()

        args = [
            "-f",
            "rotate_symmetry_axis",
            "-i",
            str(particle_path),
            "-o",
            str(self.rotated_poses_csv),
            "--poses",
            str(self.poses_csv),
            "--rotated-volume",
            str(self.rotated_particle_path),
        ]

        Plugin.runJob(self, Plugin.getSPFluoProgram(UTILS_MODULE), args=args)

    def createOuputStep(self):
        # Rotated volume
        rotated_particle = Particle.from_filename(
            self.rotated_particle_path, voxel_size=self.particle.getVoxelSize()
        )
        self._defineOutputs(**{outputs.aligned_volume.name: rotated_particle})
        self._defineRelation(RELATION_TRANSFORM, self.particle, rotated_particle)

        # Rotated particles
        output_particles = self._createSetOfParticles()

        transforms = {
            int(img_id): t for t, img_id in read_poses(self.rotated_poses_csv)
        }
        for particle in self.input_particles:
            particle: Particle

            rotated_transform = transforms[particle.getObjId()]  # new coords

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
            **{self._possibleOutputs.aligned_particles.name: output_particles}
        )
        self._defineRelation(RELATION_TRANSFORM, self.input_particles, output_particles)
