import os

from pwfluo.objects import (
    Particle,
    SetOfParticles,
)
from pwfluo.protocols import ProtFluoBase
from pyworkflow import BETA
from pyworkflow.protocol import Form, Protocol, params

from singleparticle import Plugin
from singleparticle.constants import UTILS_MODULE
from singleparticle.convert import read_poses, save_poses


class ProtSingleParticleAlignAxis(Protocol, ProtFluoBase):
    """Align the axis of a cylindrical particle"""

    OUTPUT_NAME = "aligned SetOfParticles"
    _label = "align axis"
    _devStatus = BETA
    _possibleOutputs = {OUTPUT_NAME: SetOfParticles}

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
            label="coordinates associated",
            important=True,
        )

    def _insertAllSteps(self):
        self.poses_csv = self._getExtraPath("poses.csv")
        self.rotated_poses_csv = self._getExtraPath("new_poses.csv")
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.alignStep)
        self._insertFunctionStep(self.createOuputStep)

    def prepareStep(self):
        self.input_particles: SetOfParticles = self.inputParticles.get()
        save_poses(self.poses_csv, self.input_particles)

    def alignStep(self):
        particle: Particle = self.inputParticle.get()
        particle_path = particle.getFileName()

        args = [
            "-f",
            "rotate_symmetry_axis",
            "-i",
            str(particle_path),
            "-o",
            str(self.rotated_poses_csv),
            "--poses",
            str(self.poses_csv),
        ]

        Plugin.runSPFluo(self, Plugin.getProgram(UTILS_MODULE), args=args)

    def createOuputStep(self):
        output_particles = self._createSetOfParticles()

        coords = {
            int(img_id): coord for coord, img_id in read_poses(self.rotated_poses_csv)
        }
        for particle in self.input_particles:
            particle: Particle

            rotated_coord = coords[particle.getObjId()]  # new coords

            # New file (link to particle)
            rotated_particle_path = self._getExtraPath(particle.getBaseName())
            os.link(particle.getFileName(), rotated_particle_path)

            # Creating the particle
            rotated_particle = Particle(data=rotated_particle_path)
            rotated_particle.setCoordinate3D(rotated_coord)
            rotated_particle.setImageName(particle.getImageName())
            rotated_particle.setVoxelSize(particle.getVoxelSize())
            rotated_particle.setImgId(os.path.basename(rotated_particle_path))

            output_particles.append(rotated_particle)

        output_particles.write()

        self._defineOutputs(**{self.OUTPUT_NAME: output_particles})
