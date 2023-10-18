import os
import pickle
from typing import Callable, Dict, Optional

import numpy as np
import pyworkflow.object as pwobj
from pwfluo.objects import (
    Coordinate3D,
    FluoImage,
    SetOfCoordinates3D,
    SetOfFluoImages,
)
from pwfluo.protocols import ProtFluoPicking
from pyworkflow import BETA
from pyworkflow.protocol import params

from singleparticle import Plugin
from singleparticle.constants import PICKING_MODULE, PICKING_WORKING_DIR
from singleparticle.protocols.protocol_picking_train import (
    ProtSingleParticlePickingTrain,
)


class ProtSingleParticlePickingPredict(ProtFluoPicking):
    """
    Picking for fluo data with deep learning
    """

    _label = "picking predict"
    _devStatus = BETA

    def __init__(self, **kwargs):
        ProtFluoPicking.__init__(self, **kwargs)

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        ProtFluoPicking._defineParams(self, form)
        form.addParam(
            "trainPicking",
            params.PointerParam,
            pointerClass="ProtSingleParticlePickingTrain",
            label="Train run",
            important=True,
            help="Select the train run that contains the trained PyTorch model.",
        )
        form.addParam(
            "patchSize",
            params.StringParam,
            label="Patch size",
            help="Patch size in the form 'pz py px'",
        )
        form.addParam(
            "stride",
            params.IntParam,
            label="Stride",
            default=12,
            help="Stride of the sliding window. "
            "Prefere something around patch_size/2."
            "Small values might cause Out Of Memory errors !",
        )
        form.addParam(
            "batch_size",
            params.IntParam,
            label="Batch size",
            default=1,
            help="Batch size during inference",
        )

    def _insertAllSteps(self):
        self.trainRun: ProtSingleParticlePickingTrain = self.trainPicking.get()
        self.output_dir = os.path.abspath(self._getExtraPath(PICKING_WORKING_DIR))
        self.test_dir = os.path.join(self.output_dir, "images")
        self.inputImages: SetOfFluoImages = self.inputFluoImages.get()
        self.image_paths = {}
        for im in self.inputImages:
            im: FluoImage
            im_name = im.getImgId()
            im_newPath = os.path.join(self.test_dir, im_name + ".tif")
            self.image_paths[im.getBaseName()] = os.path.basename(im_newPath)

        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.predictStep)
        self._insertFunctionStep(self.postprocessStep)
        self._insertFunctionStep(self.createOuputStep)

    def prepareStep(self):
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir, exist_ok=True)

        # Image links
        for im in self.inputImages:
            im: FluoImage
            im_path = os.path.abspath(im.getFileName())
            ext = os.path.splitext(im_path)[1]
            im_name = im.getImgId()
            im_newPath = os.path.join(self.test_dir, im_name + ".tif")
            if ext != ".tif" and ext != ".tiff":
                raise NotImplementedError(
                    f"Found ext {ext} in particles: {im_path}."
                    "Only tiff file are supported."
                )  # FIXME: allow formats accepted by AICSImageio
            else:
                os.link(im_path, im_newPath)

    def predictStep(self):
        checkpoint_path = os.path.abspath(
            self.trainRun._getExtraPath("picking", "checkpoint.pt")
        )
        ps = self.trainRun.inputCoordinates.get().getBoxSize()
        args = ["--stages", "predict"]
        args += ["--checkpoint", f"{checkpoint_path}"]
        args += ["--batch_size", f"{self.batch_size.get()}"]
        args += ["--testdir", f"{self.test_dir}"]
        args += ["--output_dir", f"{self.output_dir}"]
        args += ["--patch_size", f"{ps}"]
        args += ["--stride", f"{self.stride.get()}"]
        args += ["--extension", "tif"]
        if self.trainRun.pu.get():
            args += ["--predict_on_u_mask"]
        Plugin.runJob(self, Plugin.getSPFluoProgram(PICKING_MODULE), args=args)

    def postprocessStep(self):
        checkpoint_path = os.path.abspath(
            self.trainRun._getExtraPath("picking", "checkpoint.pt")
        )
        predictions_path = self._getExtraPath("picking", "predictions.pickle")
        args = ["--stages", "postprocess"]
        args += ["--predictions", f"{predictions_path}"]
        args += ["--checkpoint", f"{checkpoint_path}"]
        args += ["--testdir", f"{self.test_dir}"]
        args += ["--output_dir", f"{self.output_dir}"]
        args += ["--stride", f"{self.stride.get()}"]
        args += ["--extension", "tif"]
        args += ["--iterative"]
        if self.patchSize.get():
            args += ["--patch_size", f"{self.patchSize.get()}"]
        else:
            args += [
                "--patch_size",
                f"{self.trainRun.inputCoordinates.get().getBoxSize()}",
            ]
        Plugin.runJob(self, Plugin.getSPFluoProgram(PICKING_MODULE), args=args)

    def createOuputStep(self):
        for count in range(100):
            outputname = "coordinates%s" % count
            if not hasattr(self, outputname):
                suffix = "user%s" % count
                break
        suffix = self._getOutputSuffix(SetOfCoordinates3D)

        pickleFile = self._getExtraPath("picking", "predictions.pickle")
        with open(os.path.abspath(pickleFile), "rb") as f:
            preds: Dict[str, Dict[str, np.ndarray]] = pickle.load(f)

        setOfImages = self.inputImages
        step_keys = preds[
            self.image_paths[setOfImages.getFirstItem().getBaseName()]
        ].keys()
        for k in step_keys:
            if k != "raw":
                coordSet = self._createSetOfCoordinates3D(setOfImages, suffix)
                boxsize = self.trainRun.inputCoordinates.get().getBoxSize()
                coordSet.setBoxSize(boxsize)
                coordSet.setName("predCoord_" + k)
                coordSet.setVoxelSize(setOfImages.getVoxelSize())

                for image in setOfImages.iterItems():
                    if self.image_paths[image.getBaseName()] in preds:
                        pred_dict = preds[self.image_paths[image.getBaseName()]]
                        boxes = pred_dict[k]
                        readSetOfCoordinates3D(boxes, coordSet, image)
                coordSet.write()

                outputname = self.OUTPUT_PREFIX + "_" + k + "_" + suffix
                self._defineOutputs(**{outputname: coordSet})
                self._defineRelation(pwobj.RELATION_SOURCE, setOfImages, coordSet)


def readSetOfCoordinates3D(
    boxes: np.ndarray,
    coord3DSet: SetOfCoordinates3D,
    inputImage: FluoImage,
    updateItem: Optional[Callable] = None,
    scale=1,
):
    for box in boxes:
        x_min, y_min, z_min, x_max, y_max, z_max = box
        center = np.array(
            [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        )
        x, y, z = scale * center
        newCoord = Coordinate3D()
        newCoord.setFluoImage(inputImage)
        newCoord.setImageId(inputImage.getImgId())
        Lx, Ly, Lz = inputImage.getDim()
        newCoord.setPosition(z, y, x)  # FIXME which coordinate system to use ?

        # Execute Callback
        if updateItem:
            updateItem(newCoord)

        coord3DSet.enableAppend()
        coord3DSet.append(newCoord)
