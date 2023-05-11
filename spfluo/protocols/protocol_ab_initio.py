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
import glob
import os
from enum import Enum
import csv
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from pyworkflow.protocol import Protocol, params, Integer
from pyworkflow.utils import Message
from pyworkflow import BETA
import pyworkflow.object as pwobj
from pwem.emlib.image import ImageHandler
from spfluo.objects import SetOfParticles
from tomo.objects import AverageSubTomogram, SetOfSubTomograms
from tomo.protocols import ProtTomoBase


from spfluo import Plugin
from spfluo.constants import *
from spfluo.convert import convert_to_tif, getLastParticlesParams, updateSetOfSubTomograms


class outputs(Enum):
    reconstructedVolume = AverageSubTomogram
    particles = SetOfSubTomograms


class ProtSPFluoAbInitio(Protocol, ProtTomoBase):
    """
    Ab initio reconstruction
    """
    _label = 'ab initio reconstruction'
    _devStatus = BETA
    _GPU_libraries = ['no', 'cucim', 'pytorch']
    _possibleOutputs = outputs

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label="Data params")
        form.addParam('inputParticles', params.PointerParam, pointerClass='SetOfSubTomograms',
                      label="Particles", important=True,
                      help='Select the input particles.')
        form.addParam('inputPSF', params.PointerParam, pointerClass='SetOfTomograms',
                      label="PSF", important=True,
                      help='Select the PSF.')
        form.addParam('voxelSizeZ', params.FloatParam,
                      label="Voxel size Z", default=1.)
        form.addParam('voxelSizeXY', params.FloatParam,
                      label="Voxel size XY", default=1.)
        form.addParam('gpu', params.EnumParam, choices=self._GPU_libraries,
                       display=params.EnumParam.DISPLAY_LIST,
                       label='GPU Library')
        form.addSection(label="Reconstruction params")
        form.addParam('numIterMax', params.IntParam, default=20, label="Max number of epochs")
        form.addParam('N_axes', params.IntParam, default=25, label="N axes", expertLevel=params.LEVEL_ADVANCED)
        form.addParam('N_rot', params.IntParam, default=20, label="N rot", expertLevel=params.LEVEL_ADVANCED)
        form.addParam('lr', params.FloatParam, default=0.1, label="learning rate", expertLevel=params.LEVEL_ADVANCED)
    
    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self.particlesDir = os.path.abspath(self._getExtraPath("particles"))
        self.outputDir = os.path.abspath(self._getExtraPath("working_dir"))
        self.psfPath = os.path.abspath(self._getExtraPath("psf.tif"))
        self.final_reconstruction = self._getExtraPath("final_recon.mrc")
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.reconstructionStep)
        self._insertFunctionStep(self.convertOutputStep)
        self._insertFunctionStep(self.createOutputStep)
    
    def prepareStep(self):
        print('Creating particles directory')
        if not os.path.exists(self.particlesDir):
            os.makedirs(self.particlesDir, exist_ok=True)

        # Image links for particles
        particles_paths = []
        inputParticles : SetOfSubTomograms = self.inputParticles.get()
        max_dim = 0
        for im in inputParticles:
            max_dim = max(max_dim, max(im.getDim()))
            im_path = os.path.abspath(im.getFileName())
            ext = os.path.splitext(im_path)[1]
            im_name = im.getNameId()
            im_newPath = os.path.join(self.particlesDir, im_name+'.tif')
            particles_paths.append(im_newPath)
            if ext != '.tif' and ext != '.tiff':
                print(f"Convert {im_path} to TIF in {im_newPath}")
                convert_to_tif(im_path, im_newPath)
            else:
                os.link(im_path, im_newPath)

        # PSF Path
        psf = next(iter(self.inputPSF.get()))
        psf_path = os.path.abspath(psf.getFileName())
        ext = os.path.splitext(psf_path)[1]
        if ext != '.tif' and ext != '.tiff':
            print(f"Convert {psf_path} to TIF in {self.psfPath}")
            convert_to_tif(psf_path, self.psfPath)
        else:
            os.link(psf_path, self.psfPath)
        
        # Make isotropic
        input_paths = particles_paths + [self.psfPath]
        args = ["-f isotropic_resample"]
        args += ["-i"] + input_paths
        folder_isotropic = os.path.abspath(self._getExtraPath('isotropic'))
        if not os.path.exists(folder_isotropic):
            os.makedirs(folder_isotropic, exist_ok=True)
        args += [f"-o {folder_isotropic}"]
        args += [f"--spacing {self.voxelSizeZ} {self.voxelSizeXY} {self.voxelSizeXY}"]
        args = ' '.join(args)
        Plugin.runSPFluo(self, Plugin.getProgram(UTILS_MODULE), args=args)

        # Resize
        folder_resized = os.path.abspath(self._getExtraPath('isotropic_cropped'))
        if not os.path.exists(folder_resized):
            os.makedirs(folder_resized, exist_ok=True)
        input_paths = [os.path.join(folder_isotropic, f) for f in os.listdir(folder_isotropic)]
        args = ["-f resize"]
        args += ["-i"] + input_paths
        args += [f"--size {int(max_dim*2*(2**0.5))+1}"]
        args += [f"-o {folder_resized}"]
        args = ' '.join(args)
        Plugin.runSPFluo(self, Plugin.getProgram(UTILS_MODULE), args=args)

        # Links
        os.remove(self.psfPath)
        for p in particles_paths: os.remove(p)
        # Link to psf
        os.link(os.path.join(folder_resized, os.path.basename(self.psfPath)), self.psfPath)
        # Links to particles
        for p in particles_paths:
            os.link(os.path.join(folder_resized, os.path.basename(p)), p)

    def reconstructionStep(self):
        args = [
            f"--particles_dir {self.particlesDir}",
            f"--psf_path {self.psfPath}",
            f"--output_dir {self.outputDir}",
            f"--N_iter_max {self.numIterMax.get()}",
            f"--lr {self.lr.get()}",
            f"--N_axes {self.N_axes.get()}",
            f"--N_rot {self.N_rot.get()}",
        ]
        gpu = self._GPU_libraries[self.gpu.get()]
        if gpu != 'no':
            args += [f'--gpu {gpu}']
            args += ['--interp_order 1']
        args = " ".join(args)
        print("Launching reconstruction")
        Plugin.runSPFluo(self, Plugin.getProgram(AB_INITIO_MODULE), args=args)
    
    def convertOutputStep(self):
        ih = ImageHandler()
        ih.convert(os.path.join(self.outputDir, 'intermediar_results', 'final_recons.tif'), self.final_reconstruction)

    def createOutputStep(self):
        inputSetOfParticles = self.inputParticles.get()
        # Output 1 : reconstruction Volume
        reconstruction = AverageSubTomogram()
        reconstruction.setFileName(self.final_reconstruction)
        self._defineOutputs(**{outputs.reconstructedVolume.name: reconstruction})

        # Output 2 : SetOfSubTomograms
        particleParams = getLastParticlesParams(os.path.join(self.outputDir, "intermediar_results"))
        outputSetOfParticles = self._createSet(SetOfSubTomograms, 'subtomograms%s.sqlite', "particles")
        outputSetOfParticles.copyInfo(inputSetOfParticles)
        outputSetOfParticles.setCoordinates3D(inputSetOfParticles.getCoordinates3D())
        print(particleParams)
        updateSetOfSubTomograms(inputSetOfParticles, outputSetOfParticles, particleParams)
        self._defineOutputs(**{outputs.particles.name: outputSetOfParticles})
        # Transform relation
        self._defineRelation(pwobj.RELATION_TRANSFORM, inputSetOfParticles, outputSetOfParticles)
        self._defineRelation(pwobj.RELATION_SOURCE, inputSetOfParticles, outputSetOfParticles)


    
    # --------------------------- INFO functions -----------------------------------
    def _summary(self):
        """ Summarize what the protocol has done"""
        summary = []

        if self.isFinished():
            summary.append("Protocol is finished")
        return summary

    def _methods(self):
        methods = []
        return methods


class ProtSPFluoSubTomoAverage(Protocol):
    _label = 'Particle average test'
    _devStatus = BETA
    _possibleOutputs = outputs

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label="Input")
        form.addParam('inputSubTomo', params.PointerParam, pointerClass='SetOfSubTomograms',
                      label="Input Subtomo", important=True,
                      help='Select the input subtomograms.')
    
    def _insertAllSteps(self):
        self.particlesDir = os.path.abspath(self._getExtraPath("particles"))
        self.outputDir = os.path.abspath(self._getExtraPath("working_dir"))
        self.final_reconstruction = self._getExtraPath("final_recon.mrc")
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.launchStep)
        self._insertFunctionStep(self.createOuputStep)

    def prepareStep(self):
        print('Creating particles directory')
        if not os.path.exists(self.particlesDir):
            os.makedirs(self.particlesDir, exist_ok=True)
        
        inputSubTomo: SetOfSubTomograms = self.inputSubTomo.get()

        # Image links for particles
        matrices = {}
        for im in inputSubTomo:
            im_path = os.path.abspath(im.getFileName())
            ext = os.path.splitext(im_path)[1]
            im_name = im.getNameId()
            im_newPath = os.path.join(self.particlesDir, im_name+'.tif')
            if ext != '.tif' and ext != '.tiff':
                print(f"Convert {im_path} to TIF in {im_newPath}")
                convert_to_tif(im_path, im_newPath)
            else:
                os.link(im_path, im_newPath)
            
            # transformations
            matrices[im_name] = im.getTransform().getMatrix()
        
        with open(self._getExtraPath('trans_mat.pickle'), 'wb') as f:
            pickle.dump(matrices, f)

    def launchStep(self):
        args = [
            f"--particles_dir {self.particlesDir}",
            f"--transformations_path {self._getExtraPath('trans_mat.pickle')}",
            f"--average_path {self._getExtraPath('average.tif')}",
        ]
        args = " ".join(args)
        Plugin.runSPFluo(self, "python -m spfluo.ab_initio_reconstruction.tests", args=args)
    
    def createOuputStep(self):
        avrg_path = self._getExtraPath('average.tif')
        avrg_new_path = self._getExtraPath("average.mrc")
        ih = ImageHandler()
        ih.convert(avrg_path, avrg_new_path)
        average_output = AverageSubTomogram()
        average_output.setFileName(avrg_new_path)
        self._defineOutputs(**{"output": average_output})