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
import webbrowser
from enum import Enum

import numpy as np
import pyworkflow.object as pwobj
from pwfluo.objects import (
    AverageParticle,
    Particle,
    PSFModel,
    SetOfParticles,
    Transform,
)
from pwfluo.protocols import ProtFluoBase
from pyworkflow import BETA
from pyworkflow.protocol import Form, Protocol, params

from singleparticle import Plugin
from singleparticle.constants import AB_INITIO_MODULE, DEFAULT_BATCH
from singleparticle.convert import (
    getLastParticlesParams,
    save_images,
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
            "channel",
            params.IntParam,
            default=0,
            label="Reconstruct on channel?",
            help="This protocol reconstruct an average particle in one channel only.",
        )
        form.addParam(
            "gpu",
            params.BooleanParam,
            default=True,
            label="Use GPU?",
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
            "dec_prop",
            params.FloatParam,
            default=1.2,
            label="alpha r",
            expertLevel=params.LEVEL_ADVANCED,
            help="Decline rate of uniform distribution of importance sampling. "
            "A higher value means a faster convergence of the angles.",
        )

    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self.particlesDir = os.path.abspath(self._getExtraPath("particles"))
        self.outputDir = os.path.abspath(self._getExtraPath("working_dir"))
        os.makedirs(self.particlesDir, exist_ok=True)
        os.makedirs(self.outputDir, exist_ok=True)
        self.psfPath = os.path.abspath(self._getExtraPath("psf.ome.tiff"))
        self.final_reconstruction = self._getExtraPath("final_reconstruction.ome.tiff")
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.reconstructionStep)
        self._insertFunctionStep(self.createOutputStep)

    def prepareStep(self):
        # Image links for particles
        particles: SetOfParticles = self.inputParticles.get()
        psf: PSFModel = self.inputPSF.get()
        channel = (
            self.channel.get()
            if particles.getNumChannels() and particles.getNumChannels() > 0
            else None
        )

        max_dim = particles.getMaxDataSize()
        if self.pad:
            max_dim = max_dim * (2**0.5)

        # Make isotropic
        vs = particles.getVoxelSize()
        pixel_size = min(vs)
        # Downsample
        pixel_size = pixel_size * self.downsampling_factor.get()
        self.pixel_size = pixel_size

        output_dir = self._getExtraPath("isotropic_downsampled")
        os.makedirs(output_dir)
        images_paths = save_images(
            [psf] + list(particles.iterItems()),
            output_dir,
            (max_dim, max_dim, max_dim),
            channel,
            (pixel_size, pixel_size),
        )

        # Link to psf
        os.link(images_paths[0], self.psfPath)

        # Links to particles
        self.particles_paths: dict[str, str] = {}
        for p, path in zip(particles, images_paths[1:]):
            p: Particle
            self.particles_paths[p.getObjId()] = path
            os.link(path, os.path.join(self.particlesDir, os.path.basename(path)))

    def reconstructionStep(self):
        args = ["--particles_dir", f"{self.particlesDir}"]
        args += ["--psf_path", f"{self.psfPath}"]
        args += ["--output_dir", f"{self.outputDir}"]
        args += ["--N_iter_max", f"{self.numIterMax.get()}"]
        args += ["--lr", f"{self.lr.get()}"]
        args += ["--N_axes", f"{self.N_axes.get()}"]
        args += ["--N_rot", f"{self.N_rot.get()}"]
        args += ["--eps", "-100"]
        args += ["--dec_prop", self.dec_prop.get()]
        if self.gpu.get():
            args += ["--gpu"]
            args += ["--interp_order", str(1)]
            if self.minibatch.get() > 0:
                args += ["--minibatch", self.minibatch.get()]
        url = f"http://127.0.0.1:5000/{self.getProject().getShortName()}/{self.getWorkingDir().strip('Runs').strip(os.path.sep)}/dashboard.html"
        print(f"Launching reconstruction, see dashboard here: {url}")
        webbrowser.open(url)
        Plugin.runJob(self, Plugin.getSPFluoProgram(AB_INITIO_MODULE), args=args)
        os.link(
            os.path.join(self.outputDir, "final_reconstruction.ome.tiff"),
            self.final_reconstruction,
        )

    def createOutputStep(self):
        inputSetOfParticles: SetOfParticles = self.inputParticles.get()
        # Output 1 : reconstruction Volume
        reconstruction = AverageParticle.from_filename(
            self.final_reconstruction, voxel_size=(self.pixel_size, self.pixel_size)
        )
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
        outputSetOfParticles.enableAppend()
        for particle in inputSetOfParticles.iterItems():
            particle: Particle
            assert (
                particle.getObjId() in self.particles_paths
            ), f"{particle.getObjId()} not in {self.particles_paths}"
            assert (
                os.path.basename(self.particles_paths[particle.getObjId()])
                in particleParams
            ), f"{self.particles_paths[particle.getObjId()]} not in {particleParams}"
            transform = particleParams[
                os.path.basename(self.particles_paths[particle.getObjId()])
            ]
            H = np.array(transform["homogeneous_transform"])
            particle.setTransform(Transform(H))
            outputSetOfParticles.append(particle)
        outputSetOfParticles.write()

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
