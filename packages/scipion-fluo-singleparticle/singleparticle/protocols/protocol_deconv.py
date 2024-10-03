import os
from enum import Enum

import numpy as np
from pwfluo.objects import (
    FluoImage,
    Particle,
    PSFModel,
    SetOfFluoImages,
    SetOfParticles,
)
from pwfluo.protocols import ProtFluoBase
from pyworkflow import BETA
from pyworkflow.protocol import Form, Protocol, params

from singleparticle import Plugin


class ProtSingleParticleDeconvBase:
    """Base methods for deconv protocols"""

    def createGroupWidefieldParams(self, form: Form, condition: str|None=None):
        group = form.addGroup("Widefields params", condition=condition)
        group.addParam(
            "NA", params.FloatParam, label="NA", help="Numerical aperture", default=1.4
        )

        group.addParam(
            "lbda",
            params.FloatParam,
            label="λ(nm)",
            help="Wavelength in nm",
            default=540,
        )

        group.addParam(
            "ni",
            params.FloatParam,
            label="ni",
            help="Refractive index of te immersion medium",
            default=1.518,
        )

    def createNoiseParams(self, form: Form):
        group = form.addGroup("Noise params", expertLevel=params.LEVEL_ADVANCED)
        group.addParam(
            "gamma",
            params.FloatParam,
            label="Detector gain",
            help="Detector gain in electrons per analog digital unit (ADU).\n"
            "Warning: Detector gain is ignored if the standard"
            "deviation is not specified.\nLeave empty if unknown",
            allowsNull=True,
            expertLevel=params.LEVEL_ADVANCED,
        )
        group.addParam(
            "sigma",
            params.FloatParam,
            label="Readout noise",
            help="Standard deviation of the noise in e-/pixel.\n"
            "Leave empty if unknown",
            allowsNull=True,
            expertLevel=params.LEVEL_ADVANCED,
        )

    def createOptimizationParams(self, form: Form):
        group = form.addGroup("Optimization params")
        group.addParam(
            "mu",
            params.FloatParam,
            label="Log10 of the regularization level",
            default=0.0,
        )

        group.addParam(
            "maxiter",
            params.IntParam,
            label="Number of iterations",
            help="Maximum number of iterations\n" "-1 for no limit",
            default=200,
        )

        group.addParam(
            "epsilon",
            params.FloatParam,
            label="Threshold level",
            help="Threshold of hyperbolic TV",
            expertLevel=params.LEVEL_ADVANCED,
            allowsNull=True,
        )

        group.addParam(
            "nonneg",
            params.BooleanParam,
            label="Enforce nonnegativity",
            help="Enforce the positivity of the solution",
            expertLevel=params.LEVEL_ADVANCED,
            default=True,
        )

        group.addParam(
            "single",
            params.BooleanParam,
            label="Force single precision",
            expertLevel=params.LEVEL_ADVANCED,
            default=False,
        )
        return group

    def createInputParams(self, form: Form):
        group = form.addGroup("Input")
        group.addParam(
            "fluoimage",
            params.PointerParam,
            pointerClass="FluoImage",
            label="Image",
            important=True,
        )
        group.addParam(
            "channel",
            params.IntParam,
            default=0,
            label="Deconvolve on channel?",
            help="This protocol deconvolves in one channel only.",
        )
        return group


class ProtSingleParticleDeconvSetBase(ProtSingleParticleDeconvBase):
    def createInputParams(self, form: Form):
        group = form.addGroup("Input")
        group.addParam(
            "fluoimages",
            params.PointerParam,
            pointerClass="SetOfFluoImages",
            label="Images",
            important=True,
        )
        group.addParam(
            "channel",
            params.IntParam,
            default=0,
            label="Deconvolve on channel?",
            help="This protocol deconvolves in one channel only.",
        )
        return group


class ProtSingleParticleDeconv(Protocol, ProtSingleParticleDeconvBase):
    """Deconvolve an image"""

    OUTPUT_NAME = "FluoImage"
    _label = "deconvolve"
    _devStatus = BETA
    _possibleOutputs = {OUTPUT_NAME: FluoImage}

    def _defineParams(self, form: Form):
        form.addSection(label="Data params")
        self.createInputParams(form)

        form.addParam(
            "paddingMethod",
            params.StringParam,
            label="Padding size (in pixels)",
            default="30",
            help="padding in xyz directions",
            expertLevel=params.LEVEL_ADVANCED,
        )
        form.addParam(
            "usePSF",
            params.BooleanParam,
            label="PSF?",
            help="If no PSF is provided, will use the widefield params to build one.",
            default=False,
        )

        form.addParam(
            "psf",
            params.PointerParam,
            pointerClass="PSFModel",
            label="PSF",
            allowsNull=True,
            condition="usePSF is True",
        )

        self.createGroupWidefieldParams(form, condition="usePSF is False")

        form.addParam(
            "normalizePSF",
            params.BooleanParam,
            label="Normalize the PSF",
            default=False,
        )

        form.addParam(
            "crop",
            params.BooleanParam,
            label="Crop result to the same size as input",
            default=True,
        )

        form.addSection(label="Parameters")

        group = self.createOptimizationParams(form)
        group.addParam(
            "maxeval",
            params.IntParam,
            label="Number of evaluations",
            help="Maximum number of evalutions\n" "-1 for no limit",
            expertLevel=params.LEVEL_ADVANCED,
            default=-1,
        )

        self.createNoiseParams(form)

    def _insertAllSteps(self):
        self.root_dir = self._getExtraPath("rootdir")
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.prepareStepPSF)
        self._insertFunctionStep(self.deconvStep)
        self._insertFunctionStep(self.createOutputStep)

    def prepareStep(self):
        os.makedirs(self.root_dir, exist_ok=True)
        input_fluoimage: FluoImage = self.fluoimage.get()
        self.out_path = os.path.join(self.root_dir, "out.ome.tiff")

        # Input image
        a = input_fluoimage.getData().astype(np.float32)[self.channel.get()]
        self.epsilon_default_value = float(a.max()) / 1000
        self.input_float = FluoImage.from_data(
            a[None],
            os.path.join(self.root_dir, "in.ome.tiff"),
            voxel_size=input_fluoimage.getVoxelSize(),
        )

    def prepareStepPSF(self):
        self.input_psf: PSFModel | None = self.psf.get()
        self.psf_path = None
        if self.input_psf:
            a = self.input_psf.getData().astype(np.float32)
            if self.input_psf.getNumChannels() - 1 >= self.channel.get():
                channel = self.channel.get()
            else:
                channel = 0
            a = a[channel]
            self.psf_float = PSFModel.from_data(
                a[None],
                os.path.join(self.root_dir, "psf.ome.tiff"),
                voxel_size=self.input_psf.getVoxelSize(),
            )

    def deconvStep(self):
        args = list(
            map(os.path.abspath, [self.input_float.getFileName(), self.out_path])
        )
        if self.usePSF.get():
            args += ["-dxy", f"{self.input_psf.getVoxelSize()[0]*1000}"]
            args += ["-dz", f"{self.input_psf.getVoxelSize()[1]*1000}"]
            args += ["-psf", f"{os.path.abspath(self.psf_float.getFileName())}"]
        else:
            args += ["-dxy", f"{self.input_float.getVoxelSize()[0]*1000}"]
            args += ["-dz", f"{self.input_float.getVoxelSize()[1]*1000}"]
            args += ["-NA", f"{self.NA.get()}"]
            args += ["-lambda", f"{self.lbda.get()}"]
            args += ["-ni", f"{self.ni.get()}"]
        if self.normalizePSF.get():
            args += ["-normalize"]
        if self.sigma.get():
            args += ["-noise", f"{self.sigma.get()}"]
        if self.gamma.get():
            args += ["-gain", f"{self.gamma.get()}"]
        args += ["-mu", f"{10**self.mu.get()}"]
        eps = self.epsilon.get()
        args += ["-epsilon", f"{eps if eps else self.epsilon_default_value}"]
        if self.nonneg.get():
            args += ["-min", "0"]
        if self.single.get():
            args += ["-single"]
        args += ["-maxiter", f"{self.maxiter.get()}"]
        args += ["-maxeval", f"{self.maxeval.get()}"]
        args += ["-pad", f"{self.paddingMethod.get()}"]
        args += ["-debug", "-verbose"]
        if self.crop.get():
            args += ["-crop"]

        Plugin.runJob(self, Plugin.getMicroTipiProgram("deconv"), args=args)

    def createOutputStep(self):
        deconv_im = FluoImage.from_filename(
            self.out_path, voxel_size=self.input_float.getVoxelSize()
        )
        deconv_im.setImgId(self.input_float.getImgId())
        deconv_im.cleanObjId()
        self._defineOutputs(**{self.OUTPUT_NAME: deconv_im})


class outputs(Enum):
    psf = PSFModel
    deconvolution = FluoImage


class ProtSingleParticleBlindDeconv(Protocol, ProtSingleParticleDeconvBase):
    """Deconvolve an image"""

    OUTPUT_NAME = "FluoImage"
    _label = "blind deconvolve"
    _devStatus = BETA
    _possibleOutputs = outputs

    WEIGHTINGS = [("variance=1", "CONSTANT"), ("computed var map", "COMPUTED_VAR_MAP")]

    def _defineParams(self, form: Form):
        form.addSection(label="Data params")
        self.createInputParams(form)

        form.addParam(
            "paddingMethod",
            params.StringParam,
            label="Padding size (in pixels)",
            default="30",
            help="padding in xyz directions",
            expertLevel=params.LEVEL_ADVANCED,
        )

        self.createGroupWidefieldParams(form)

        form.addParam(
            "crop",
            params.BooleanParam,
            label="Crop result to the same size as input",
            default=True,
        )

        form.addSection(label="Parameters")

        form.addParam(
            "nbloops",
            params.IntParam,
            label="number of loops",
            default=2,
            help="The number of loops of the algorithm\n"
            "The higher, the potentially longer ",
        )

        self.createNoiseParams(form)

        group = self.createOptimizationParams(form)
        group.addParam(
            "maxIterDefocus",
            params.IntParam,
            label="Max nb. of iterations for defocus",
            default=20,
            expertLevel=params.LEVEL_ADVANCED,
        )

        group.addParam(
            "maxIterPhase",
            params.IntParam,
            label="Max nb. of iterations for phase",
            default=20,
            expertLevel=params.LEVEL_ADVANCED,
        )

        group.addParam(
            "maxIterModulus",
            params.IntParam,
            label="Max nb. of iterations for modulus",
            default=0,
            expertLevel=params.LEVEL_ADVANCED,
        )

        group = form.addGroup("PSF model", expertLevel=params.LEVEL_ADVANCED)
        group.addParam(
            "nPhase",
            params.IntParam,
            label="Number of phase coefs Nα",
            help="Number of zernike describing the pupil phase",
            default=19,
            expertLevel=params.LEVEL_ADVANCED,
        )

        group.addParam(
            "nModulus",
            params.IntParam,
            label="Number of modulus coefs Nβ",
            help="Number of zernike describing the pupil modulus",
            default=0,
            expertLevel=params.LEVEL_ADVANCED,
        )

        group.addParam(
            "radial",
            params.BooleanParam,
            label="Radially symmetric PSF",
            default=False,
            expertLevel=params.LEVEL_ADVANCED,
        )

    def _insertAllSteps(self):
        self.root_dir = self._getExtraPath("rootdir")
        self.out_psf_path = os.path.join(self.root_dir, "psf.ome.tiff")
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.deconvStep)
        self._insertFunctionStep(self.createOutputStep)

    def prepareStep(self):
        os.makedirs(self.root_dir, exist_ok=True)
        input_fluoimage: FluoImage = self.fluoimage.get()
        self.out_path = os.path.join(self.root_dir, "out.ome.tiff")

        # Input image
        a = input_fluoimage.getData().astype(np.float32)[self.channel.get()]
        self.epsilon_default_value = float(a.max()) / 1000
        self.input_float = FluoImage.from_data(
            a[None],
            os.path.join(self.root_dir, "in.ome.tiff"),
            voxel_size=input_fluoimage.getVoxelSize(),
        )

    def deconvStep(self):
        args = list(
            map(
                os.path.abspath,
                [self.input_float.getFileName(), self.out_path, self.out_psf_path],
            )
        )
        args += ["-dxy", f"{self.input_float.getVoxelSize()[0]*1000}"]
        args += ["-dz", f"{self.input_float.getVoxelSize()[1]*1000}"]
        # Widefield params
        args += ["-NA", f"{self.NA.get()}"]
        args += ["-lambda", f"{self.lbda.get()}"]
        args += ["-ni", f"{self.ni.get()}"]
        args += ["-nbloops", f"{self.nbloops.get()}"]
        args += ["-nPhase", f"{self.nPhase.get()}"]
        args += ["-nModulus", f"{self.nModulus.get()}"]
        if self.radial.get():
            args += ["-radial"]
        args += ["-maxIterDefocus", f"{self.maxIterDefocus.get()}"]
        args += ["-maxIterPhase", f"{self.maxIterPhase.get()}"]
        args += ["-maxIterModulus", f"{self.maxIterModulus.get()}"]

        if self.crop.get():
            args += ["-crop"]

        # Parameters
        if self.gamma.get():
            args += ["-gain", f"{self.gamma.get()}"]
        if self.sigma.get():
            args += ["-readoutNoise", f"{self.sigma.get()}"]
        args += ["-mu", f"{10**self.mu.get()}"]
        args += ["-nbIterDeconv", f"{self.maxiter.get()}"]
        eps = self.epsilon.get()
        args += ["-epsilon", f"{eps if eps else self.epsilon_default_value}"]
        args += ["-debug"]
        args += ["-pad", f"{self.paddingMethod.get()}"]
        if not self.nonneg.get():
            args += ["-negativity"]
        if self.single.get():
            args += ["-single"]
        Plugin.runJob(self, Plugin.getMicroTipiProgram("blinddeconv"), args=args)

    def createOutputStep(self):
        deconv_im = FluoImage.from_filename(
            self.out_path, voxel_size=self.input_float.getVoxelSize()
        )
        deconv_im.setImgId(self.input_float.getImgId())
        deconv_im.cleanObjId()
        self._defineOutputs(**{outputs.deconvolution.name: deconv_im})

        psf = PSFModel.from_filename(
            self.out_psf_path, voxel_size=self.input_float.getVoxelSize()
        )
        self._defineOutputs(**{outputs.psf.name: psf})


class ProtSingleParticleDeconvSet(
    Protocol, ProtSingleParticleDeconvSetBase, ProtFluoBase
):
    """Deconvolve a set of images"""

    OUTPUT_NAME = "deconvolved images"
    _label = "deconvolve set"
    OUTPUT_PREFIX = "Deconv"
    _devStatus = BETA
    _possibleOutputs = {OUTPUT_NAME: SetOfFluoImages}

    def _defineParams(self, form: Form):
        form.addSection(label="Data params")
        self.createInputParams(form)

        form.addParam(
            "paddingMethod",
            params.StringParam,
            label="Padding size (in pixels)",
            default="30",
            help="padding in xyz directions",
            expertLevel=params.LEVEL_ADVANCED,
        )
        form.addParam(
            "usePSF",
            params.BooleanParam,
            label="PSF?",
            help="If no PSF is provided, will use the widefield params to build one.",
            default=False,
        )

        form.addParam(
            "psf",
            params.PointerParam,
            pointerClass="PSFModel",
            label="PSF",
            allowsNull=True,
            condition="usePSF is True",
        )

        self.createGroupWidefieldParams(form, condition="usePSF is False")

        form.addParam(
            "normalizePSF",
            params.BooleanParam,
            label="Normalize the PSF",
            default=False,
        )

        form.addParam(
            "crop",
            params.BooleanParam,
            label="Crop result to the same size as input",
            default=True,
        )

        form.addSection(label="Parameters")

        group = self.createOptimizationParams(form)
        group.addParam(
            "maxeval",
            params.IntParam,
            label="Number of evaluations",
            help="Maximum number of evalutions\n" "-1 for no limit",
            expertLevel=params.LEVEL_ADVANCED,
            default=-1,
        )

        self.createNoiseParams(form)

    def _insertAllSteps(self):
        self.root_dir = self._getExtraPath("rootdir")
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.prepareStepPSF)
        self._insertFunctionStep(self.deconvStep)
        self._insertFunctionStep(self.createOutputStep)

    def prepareStep(self):
        os.makedirs(self.root_dir, exist_ok=True)
        input_fluoimages: SetOfFluoImages = self.fluoimages.get()
        self.fluoimages_dir = os.path.join(self.root_dir, "fluoimages")
        self.fluoimages_deconv_dir = os.path.join(self.root_dir, "fluoimages_deconv")
        os.makedirs(self.fluoimages_dir)
        os.makedirs(self.fluoimages_deconv_dir)
        self.out_path = os.path.join(self.root_dir, "out.ome.tiff")

        self.input_fluoimages_float: list[FluoImage] = []

        # Input image
        for fluoimage in input_fluoimages:
            fluoimage: FluoImage
            a = fluoimage.getData().astype(np.float32)[self.channel.get()]
            self.epsilon_default_value = float(a.max()) / 1000
            self.input_fluoimages_float.append(
                FluoImage.from_data(
                    a[None],
                    os.path.join(self.fluoimages_dir, fluoimage.getBaseName()),
                    voxel_size=fluoimage.getVoxelSize(),
                )
            )

    def prepareStepPSF(self):
        self.input_psf: PSFModel | None = self.psf.get()
        self.psf_path = None
        if self.input_psf:
            a = self.input_psf.getData().astype(np.float32)
            if self.input_psf.getNumChannels() - 1 >= self.channel.get():
                channel = self.channel.get()
            else:
                channel = 0
            a = a[channel]
            self.psf_float = PSFModel.from_data(
                a[None],
                os.path.join(self.root_dir, "psf.ome.tiff"),
                voxel_size=self.input_psf.getVoxelSize(),
            )

    def deconvStep(self):
        for input_float in self.input_fluoimages_float:
            args = list(
                map(
                    os.path.abspath,
                    [
                        input_float.getFileName(),
                        os.path.join(
                            self.fluoimages_deconv_dir, input_float.getBaseName()
                        ),
                    ],
                )
            )
            if self.usePSF.get():
                args += ["-dxy", f"{self.input_psf.getVoxelSize()[0]*1000}"]
                args += ["-dz", f"{self.input_psf.getVoxelSize()[1]*1000}"]
                args += ["-psf", f"{os.path.abspath(self.psf_float.getFileName())}"]
            else:
                args += ["-dxy", f"{input_float.getVoxelSize()[0]*1000}"]
                args += ["-dz", f"{input_float.getVoxelSize()[1]*1000}"]
                args += ["-NA", f"{self.NA.get()}"]
                args += ["-lambda", f"{self.lbda.get()}"]
                args += ["-ni", f"{self.ni.get()}"]
            if self.normalizePSF.get():
                args += ["-normalize"]
            if self.sigma.get():
                args += ["-noise", f"{self.sigma.get()}"]
            if self.gamma.get():
                args += ["-gain", f"{self.gamma.get()}"]
            args += ["-mu", f"{10**self.mu.get()}"]
            eps = self.epsilon.get()
            args += ["-epsilon", f"{eps if eps else self.epsilon_default_value}"]
            if self.nonneg.get():
                args += ["-min", "0"]
            if self.single.get():
                args += ["-single"]
            args += ["-maxiter", f"{self.maxiter.get()}"]
            args += ["-maxeval", f"{self.maxeval.get()}"]
            args += ["-pad", f"{self.paddingMethod.get()}"]
            args += ["-debug", "-verbose"]
            if self.crop.get():
                args += ["-crop"]

            Plugin.runJob(self, Plugin.getMicroTipiProgram("deconv"), args=args)

    def createOutputStep(self):
        self.inputSet: SetOfFluoImages = self.fluoimages.get()
        if isinstance(self.inputSet, SetOfParticles):
            self.outputSet = self._createSetOfParticles(
                self._getOutputSuffix(SetOfParticles)
            )
            self.outputSet.enableAppend()
            for particle in self.inputSet.iterItems():
                particle: Particle
                particle_deconv_path = os.path.join(
                    self.fluoimages_deconv_dir, particle.getBaseName()
                )
                self.outputSet.append(
                    Particle.from_filename(
                        particle_deconv_path, voxel_size=particle.getVoxelSize()
                    )
                )
            self.outputSet.write()
        self._defineOutputs(**{self.OUTPUT_NAME: self.outputSet})
