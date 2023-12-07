import os
from enum import Enum

import numpy as np
import tifffile
from pwfluo.objects import FluoImage, PSFModel
from pwfluo.protocols import ProtFluoBase
from pyworkflow import BETA
from pyworkflow.protocol import Form, Protocol, params

from singleparticle import Plugin


class ProtSingleParticleDeconv(Protocol, ProtFluoBase):
    """Deconvolve an image"""

    OUTPUT_NAME = "FluoImage"
    _label = "deconvolve"
    _devStatus = BETA
    _possibleOutputs = {OUTPUT_NAME: FluoImage}

    def _defineParams(self, form: Form):
        form.addSection(label="Data params")
        group = form.addGroup("Input")
        group.addParam(
            "fluoimage",
            params.PointerParam,
            pointerClass="FluoImage",
            label="Image",
            important=True,
        )

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

        group = form.addGroup("Widefields params", condition="usePSF is False")
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

        group.addParam(
            "maxeval",
            params.IntParam,
            label="Number of evaluations",
            help="Maximum number of evalutions\n" "-1 for no limit",
            expertLevel=params.LEVEL_ADVANCED,
            default=-1,
        )

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

    def _insertAllSteps(self):
        self.root_dir = self._getExtraPath("rootdir")
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.deconvStep)
        self._insertFunctionStep(self.createOutputStep)

    def prepareStep(self):
        os.makedirs(self.root_dir, exist_ok=True)
        self.input_fluoimage: FluoImage = self.fluoimage.get()
        self.input_psf: PSFModel | None = self.psf.get()
        self.in_path = os.path.join(self.root_dir, "in.ome.tiff")
        self.out_path = os.path.join(self.root_dir, "out.ome.tiff")
        self.psf_path = None

        # Input image

        a = self.input_fluoimage.getData().astype(np.float64)
        a = (a - a.min()) / (a.max() - a.min())
        tifffile.imwrite(
            self.in_path,
            a,
            metadata={"axes": self.input_fluoimage.img.dims.order},
        )
        self.epsilon_default_value = float(a.max()) / 1000

        # Input PSF
        if self.input_psf:
            self.psf_path = os.path.join(self.root_dir, "psf.ome.tiff")
            a = self.input_psf.getData().astype(np.float64)
            a = (a - a.min()) / (a.max() - a.min())
            tifffile.imwrite(
                self.psf_path,
                a,
                metadata={"axes": self.input_psf.img.dims.order},
            )

    def deconvStep(self):
        args = list(map(os.path.abspath, [self.in_path, self.out_path]))
        if self.usePSF.get():
            args += ["-dxy", f"{self.input_psf.getVoxelSize()[0]*1000}"]
            args += ["-dz", f"{self.input_psf.getVoxelSize()[1]*1000}"]
            args += ["-psf", f"{os.path.abspath(self.psf_path)}"]
        else:
            args += ["-dxy", f"{self.input_fluoimage.getVoxelSize()[0]*1000}"]
            args += ["-dz", f"{self.input_fluoimage.getVoxelSize()[1]*1000}"]
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
        deconv_im = FluoImage(data=self.out_path)
        deconv_im.setVoxelSize(self.input_fluoimage.getVoxelSize())
        deconv_im.setImgId(self.input_fluoimage.getImgId())
        deconv_im.cleanObjId()
        self._defineOutputs(**{self.OUTPUT_NAME: deconv_im})


class outputs(Enum):
    psf = PSFModel
    deconvolution = FluoImage


class ProtSingleParticleBlindDeconv(Protocol, ProtFluoBase):
    """Deconvolve an image"""

    OUTPUT_NAME = "FluoImage"
    _label = "blind deconvolve"
    _devStatus = BETA
    _possibleOutputs = outputs

    WEIGHTINGS = ["variance=1", "computed var map"]

    def _defineParams(self, form: Form):
        form.addSection(label="Data params")
        group = form.addGroup("Input")
        group.addParam(
            "fluoimage",
            params.PointerParam,
            pointerClass="FluoImage",
            label="Image",
            important=True,
        )

        form.addParam(
            "paddingMethod",
            params.StringParam,
            label="Padding size (in pixels)",
            default="30",
            help="padding in xyz directions",
            expertLevel=params.LEVEL_ADVANCED,
        )

        group = form.addGroup("Widefield params")
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
            help="Refractive index of the immersion medium",
            default=1.518,
        )

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

        group = form.addGroup("Noise params")
        group.addParam(
            "weighting",
            params.EnumParam,
            choices=self.WEIGHTINGS,
            label="Weighting method",
            display=params.EnumParam.DISPLAY_COMBO,
            default=1,
        )
        group.addParam(
            "gamma",
            params.FloatParam,
            label="Detector gain",
            help="Detector gain in electrons per analog digital unit (ADU).",
            condition="weighting==1 or weighting==2",
            allowsNull=True,
        )
        group.addParam(
            "sigma",
            params.FloatParam,
            label="Readout noise",
            help="Standard deviation of the noise in e-/pixel.",
            condition="weighting==1 or weighting==2",
            allowsNull=True,
        )

        group = form.addGroup("Optimization params")
        group.addParam(
            "mu",
            params.FloatParam,
            label="Log10 of the regularization level",
            default=0,
        )

        group.addParam(
            "maxiter",
            params.IntParam,
            label="Number of iterations",
            help="Maximum number of iterations\n" "-1 for no limit",
            default=50,
        )

        group.addParam(
            "epsilon",
            params.FloatParam,
            label="Threshold level",
            help="Threshold of hyperbolic TV",
            expertLevel=params.LEVEL_ADVANCED,
            default=0.01,
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
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.deconvStep)
        self._insertFunctionStep(self.createOutputStep)

    def prepareStep(self):
        os.makedirs(self.root_dir, exist_ok=True)
        self.input_fluoimage: FluoImage = self.fluoimage.get()
        self.in_path = os.path.join(self.root_dir, "in.ome.tiff")
        self.out_path = os.path.join(self.root_dir, "out.ome.tiff")
        self.out_psf_path = os.path.join(self.root_dir, "psf.ome.tiff")
        a = self.input_fluoimage.getData().astype(np.float64)
        a = (a - a.min()) / (a.max() - a.min())
        tifffile.imwrite(self.in_path, a)
        self.epsilon_default_value = float(a.max()) / 1000

    def deconvStep(self):
        args = list(
            map(os.path.abspath, [self.in_path, self.out_path, self.out_psf_path])
        )
        args += ["-dxy", f"{self.input_fluoimage.getVoxelSize()[0]*1000}"]
        args += ["-dz", f"{self.input_fluoimage.getVoxelSize()[1]*1000}"]
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
        args += [
            "-weighting",
            f"{self.WEIGHTINGS[self.weighting.get()].upper().replace(' ','_')}",
        ]
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
        deconv_im = FluoImage(data=self.out_path)
        deconv_im.setVoxelSize(self.input_fluoimage.getVoxelSize())
        deconv_im.setImgId(self.input_fluoimage.getImgId())
        deconv_im.cleanObjId()
        self._defineOutputs(**{outputs.deconvolution.name: deconv_im})

        psf = PSFModel(data=self.out_psf_path)
        psf.setVoxelSize(self.input_fluoimage.getVoxelSize())
        self._defineOutputs(**{outputs.psf.name: psf})
