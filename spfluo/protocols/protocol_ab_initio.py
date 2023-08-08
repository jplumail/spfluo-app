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
import pickle
from enum import Enum

import pyworkflow.object as pwobj
from pwfluo.objects import AverageParticle, PSFModel, SetOfParticles
from pwfluo.protocols import ProtFluoBase
from pyworkflow import BETA
from pyworkflow.protocol import Form, Protocol, params

from spfluo import Plugin
from spfluo.constants import AB_INITIO_MODULE, UTILS_MODULE
from spfluo.convert import (
    getLastParticlesParams,
    save_particles,
    save_psf,
    updateSetOfParticles,
)


class outputs(Enum):
    reconstructedVolume = AverageParticle
    particles = SetOfParticles


class ProtSPFluoAbInitio(Protocol, ProtFluoBase):
    """
    Ab initio reconstruction
    """

    _label = "ab initio reconstruction"
    _devStatus = BETA
    _GPU_libraries = ["no", "cucim", "pytorch"]
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
            params.EnumParam,
            choices=self._GPU_libraries,
            display=params.EnumParam.DISPLAY_LIST,
            label="GPU Library",
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
            "numIterMax", params.IntParam, default=20, label="Max number of epochs"
        )
        form.addParam(
            "N_axes",
            params.IntParam,
            default=25,
            label="N axes",
            expertLevel=params.LEVEL_ADVANCED,
        )
        form.addParam(
            "N_rot",
            params.IntParam,
            default=20,
            label="N rot",
            expertLevel=params.LEVEL_ADVANCED,
        )
        form.addParam(
            "lr",
            params.FloatParam,
            default=0.1,
            label="learning rate",
            expertLevel=params.LEVEL_ADVANCED,
        )
        form.addParam(
            "eps",
            params.FloatParam,
            default=-100,
            label="eps",
            expertLevel=params.LEVEL_ADVANCED,
            help="minimum gain in energy before stopping",
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
            max_dim = int(max_dim * 2 * (2**0.5)) + 1
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
        gpu = self._GPU_libraries[self.gpu.get()]
        if gpu != "no":
            args += ["--gpu", gpu]
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


class ProtSPFluoParticleAverage(Protocol):
    _label = "Particle average test"
    _devStatus = BETA
    _possibleOutputs = outputs

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label="Input")
        form.addParam(
            "inputParticle",
            params.PointerParam,
            pointerClass="SetOfParticles",
            label="Input Particle",
            important=True,
            help="Select the input particles.",
        )

    def _insertAllSteps(self):
        self.particlesDir = os.path.abspath(self._getExtraPath("particles"))
        self.outputDir = os.path.abspath(self._getExtraPath("working_dir"))
        self.final_reconstruction = self._getExtraPath("final_recon.tif")
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.launchStep)
        self._insertFunctionStep(self.createOuputStep)

    def prepareStep(self):
        print("Creating particles directory")
        if not os.path.exists(self.particlesDir):
            os.makedirs(self.particlesDir, exist_ok=True)

        inputParticles: SetOfParticles = self.inputParticle.get()

        # Image links for particles
        matrices = {}
        for im in inputParticles:
            im_path = os.path.abspath(im.getFileName())
            ext = os.path.splitext(im_path)[1]
            im_name = im.strId()
            im_newPath = os.path.join(self.particlesDir, im_name + ".tif")
            if ext != ".tif" and ext != ".tiff":
                raise NotImplementedError(
                    f"Found ext {ext} in particles: {im_path}. "
                    "Only tiff file are supported."
                )
            else:
                os.link(im_path, im_newPath)

            # transformations
            matrices[im_name] = im.getTransform().getMatrix()

        with open(self._getExtraPath("trans_mat.pickle"), "wb") as f:
            pickle.dump(matrices, f)

    def launchStep(self):
        args = ["--particles_dir", f"{self.particlesDir}"]
        args += ["--transformations_path", f"{self._getExtraPath('trans_mat.pickle')}"]
        args += ["--average_path", f"{self._getExtraPath('average.tif')}"]
        Plugin.runSPFluo(
            self, "python -m spfluo.ab_initio_reconstruction.tests", args=args
        )

    def createOuputStep(self):
        avrg_path = self._getExtraPath("average.tif")
        average_output = AverageParticle(data=avrg_path)
        self._defineOutputs(**{"output": average_output})
