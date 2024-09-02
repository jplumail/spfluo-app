import os
from os.path import abspath, basename, join

from pwfluo.objects import FluoImage, SetOfFluoImages
from pwfluo.protocols import ProtFluoBase
from pyworkflow import BETA
from pyworkflow.protocol import Form, Protocol
from pyworkflow.protocol.params import EnumParam, FloatParam, PointerParam

from singleparticle import Plugin
from singleparticle.constants import UTILS_MODULE


class ProtSingleParticleUtils(Protocol, ProtFluoBase):
    """
    Use SPFluo utils functions.
    """

    OUTPUT_NAME = "FluoImages"

    _label = "utils"
    _devStatus = BETA

    FUNCTION_CHOICES = ["resample"]

    def _defineParams(self, form: Form):
        form.addSection(label="Input")
        form.addParam(
            "inputFluoImages",
            PointerParam,
            label="Input Images",
            important=True,
            pointerClass=SetOfFluoImages,
            help="Select the Image to be used during picking.",
        )
        form.addParam(
            "function",
            EnumParam,
            choices=self.FUNCTION_CHOICES,
            display=EnumParam.DISPLAY_COMBO,
            label="Function",
        )
        form.addParam(
            "factor",
            FloatParam,
            condition=f"function=={self.FUNCTION_CHOICES.index('resample')}",
            label="Resampling Factor",
        )

    def _insertAllSteps(self):
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.functionStep)
        self._insertFunctionStep(self.outputStep)

    def prepareStep(self):
        self.input_images = []
        self.output_dir = self._getExtraPath("output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.inputfluoImages: SetOfFluoImages = self.inputFluoImages.get()
        for image in self.inputfluoImages:
            image: FluoImage
            self.input_images.append(abspath(image.getFileName()))

    def functionStep(self):
        args = ["--input", *self.input_images, "--output", self.output_dir]
        args += ["--function", self.FUNCTION_CHOICES[self.function.get()]]
        if self.FUNCTION_CHOICES[self.function.get()] == "resample":
            args += ["--factor", str(self.factor.get())]

        Plugin.runJob(self, Plugin.getSPFluoProgram(UTILS_MODULE), args=args)

    def outputStep(self):
        imgSet = self._createSetOfFluoImages()
        vs = self.inputfluoImages.getVoxelSize()
        if self.function.get() == "resample":
            vs = (vs[0] * self.factor.get(), vs[1] * self.factor.get())
        imgSet.setVoxelSize(vs)

        for input_im in self.inputfluoImages:
            input_im: FluoImage
            output_path = join(self.output_dir, basename(input_im.getFileName()))
            assert os.path.exists(output_path)
            im = FluoImage.from_filename(
                output_path, voxel_size=vs, num_channels=input_im.getNumChannels()
            )
            im.setImgId(input_im.getImgId())
            im.cleanObjId()
            imgSet.append(im)
        imgSet.write()

        self._defineOutputs(**{self.OUTPUT_NAME: imgSet})
