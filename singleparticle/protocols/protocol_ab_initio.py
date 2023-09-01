# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     you (you@yourinstitution.email)
# *
# * your institution
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'you@yourinstitution.email'
# *
# **************************************************************************


"""
Describe your python module here:
This module will provide the traditional Hello world example
"""
import os
from enum import Enum

import pyworkflow.object as pwobj
from pwfluo.objects import AverageParticle, PSFModel, SetOfParticles
from pwfluo.protocols import ProtFluoBase
from pyworkflow import BETA
from pyworkflow.protocol import Form, Protocol, params

from singleparticle import Plugin
from singleparticle.constants import AB_INITIO_MODULE, UTILS_MODULE
from singleparticle.convert import (
    getLastParticlesParams,
    save_particles,
    save_psf,
    updateSetOfParticles,
)


class outputs(Enum):
    reconstructedVolume = AverageParticle
    particles = SetOfParticles


class ProtSingleParticleAbInitio(Protocol, ProtFluoBase):
    """
    Ab initio reconstruction
    """

    _label = "ab initio reconstruction"
    _devStatus = BETA
    _possibleOutputs = outputs

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form: Form):
        form.addSection(label="Data params")
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
            label="Use GPU?",
        )
        form.addParam(
            "pad",
            params.BooleanParam,
            default=True,
            expertLevel=params.LEVEL_ADVANCED,
            label="Pad particles?",
            help="Disable only if all of your particles have enough blank space around"
            "them.",
        )
        form.addSection(label="Reconstruction params")
        form.addParam(
            "numIterMax",
            params.IntParam,
            default=20,
            label="Max number of epochs",
            help="The algorithm will perform better with more epochs."
            "It can stop earlier, if it stagnates."
            "See the eps parameter in advanced mode.",
        )
        form.addParam(
            "N_axes",
            params.IntParam,
            default=25,
            label="N axes",
            expertLevel=params.LEVEL_ADVANCED,
            help="N_axes*N_rot is the number of rotations the algorithm"
            "will test at each epoch."
            "Increasing this parameter can lead to longer epochs"
            "and to out of memory errors.",
        )
        form.addParam(
            "N_rot",
            params.IntParam,
            default=20,
            label="N rot",
            expertLevel=params.LEVEL_ADVANCED,
            help="N_axes*N_rot is the number of rotations the algorithm will test."
            "Increasing this parameter can lead to longer epochs"
            "and to out of memory errors.",
        )
        form.addParam(
            "lr",
            params.FloatParam,
            default=0.1,
            label="learning rate",
            expertLevel=params.LEVEL_ADVANCED,
            help="Increase to get faster results. "
            "However, a value too high can break the algorithm.",
        )
        form.addParam(
            "eps",
            params.FloatParam,
            default=-100,
            label="eps",
            expertLevel=params.LEVEL_ADVANCED,
            help="minimum gain in energy before stopping"
            "a negative value allows for eventual losses in energy",
        )

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self.particlesDir = os.path.abspath(self._getExtraPath("particles"))
        self.outputDir = os.path.abspath(self._getExtraPath("working_dir"))
        self.psfPath = os.path.abspath(self._getExtraPath("psf.tif"))
        self.final_reconstruction = self._getExtraPath("final_reconstruction.tif")
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.reconstructionStep)
        self._insertFunctionStep(self.createOutputStep)

    def prepareStep(self):
        # Image links for particles
        particles: SetOfParticles = self.inputParticles.get()
        channel = self.channel.get() if particles.getNumChannels() > 0 else None
        particles_paths, max_dim = save_particles(
            self.particlesDir, particles, channel=channel
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
        args = ["--particles_dir", f"{self.particlesDir}"]
        args += ["--psf_path", f"{self.psfPath}"]
        args += ["--output_dir", f"{self.outputDir}"]
        args += ["--N_iter_max", f"{self.numIterMax.get()}"]
        args += ["--lr", f"{self.lr.get()}"]
        args += ["--N_axes", f"{self.N_axes.get()}"]
        args += ["--N_rot", f"{self.N_rot.get()}"]
        args += ["--eps", self.eps.get()]
        if self.gpu.get():
            args += ["--gpu", "pytorch"]
            args += ["--interp_order", str(1)]
        print("Launching reconstruction")
        Plugin.runSPFluo(self, Plugin.getProgram(AB_INITIO_MODULE), args=args)
        os.link(
            os.path.join(self.outputDir, "final_recons.tif"),
            self.final_reconstruction,
        )

    def createOutputStep(self):
        inputSetOfParticles = self.inputParticles.get()
        # Output 1 : reconstruction Volume
        reconstruction = AverageParticle()
        reconstruction.setFileName(self.final_reconstruction)
        self._defineOutputs(**{outputs.reconstructedVolume.name: reconstruction})

        # Output 2 : SetOfParticles
        particleParams = getLastParticlesParams(
            os.path.join(self.outputDir, "intermediar_results")
        )
        outputSetOfParticles = self._createSet(
            SetOfParticles, "particles%s.sqlite", "particles"
        )
        outputSetOfParticles.copyInfo(inputSetOfParticles)
        outputSetOfParticles.setCoordinates3D(inputSetOfParticles.getCoordinates3D())
        updateSetOfParticles(inputSetOfParticles, outputSetOfParticles, particleParams)
        self._defineOutputs(**{outputs.particles.name: outputSetOfParticles})
        # Transform relation
        self._defineRelation(
            pwobj.RELATION_TRANSFORM, inputSetOfParticles, outputSetOfParticles
        )
        self._defineRelation(
            pwobj.RELATION_SOURCE, inputSetOfParticles, outputSetOfParticles
        )

    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        """Summarize what the protocol has done"""
        summary = []

        if self.isFinished():
            summary.append("Protocol is finished")
        return summary

    def _methods(self):
        methods = []
        return methods
