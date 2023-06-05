import os
import pickle
from typing import Callable, Dict, Optional
import numpy as np
from pyworkflow import BETA
import pyworkflow.object as pwobj
from pyworkflow.protocol import params
from spfluo.objects.data import (
    Coordinate3D,
    FluoImage,
    SetOfCoordinates3D,
    SetOfFluoImages,
)
from .protocol_base import ProtFluoPicking
from spfluo import Plugin
from spfluo.constants import *
from spfluo.protocols.protocol_picking_train import ProtSPFluoPickingTrain


class ProtSPFluoPickingPredict(ProtFluoPicking):
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
            pointerClass="ProtSPFluoPickingTrain",
            label="Train run",
            important=True,
            help="Select the train run that contains the trained PyTorch model.",
        )
        # form.addParam('inputImages', params.PointerParam, pointerClass='SetOfTomograms',
        #              label="Images", important=True,
        #              help='Select the input images.')
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
            help="Stride of the sliding window. Prefere something around patch_size/2. Small values might cause Out Of Memory errors !",
        )
        form.addParam(
            "batch_size",
            params.IntParam,
            label="Batch size",
            default=1,
            help="Batch size during inference",
        )

    def _insertAllSteps(self):
        self.trainRun: ProtSPFluoPickingTrain = self.trainPicking.get()
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
                    f"Found ext {ext} in particles: {im_path}. Only tiff file are supported."
                )  # FIXME: allow formats accepted by AICSImageio
            else:
                os.link(im_path, im_newPath)

    def predictStep(self):
        args = [
            "--stages predict",
            f"--checkpoint {os.path.abspath(self.trainRun._getExtraPath('picking', 'checkpoint.pt'))}",
            f"--batch_size {self.batch_size.get()}",
            f"--testdir {self.test_dir}",
            f"--output_dir {self.output_dir}",
            f"--patch_size {self.trainRun.inputCoordinates.get().getBoxSize()}",
            f"--stride {self.stride.get()}",
            "--extension tif",
        ]
        if self.trainRun.pu.get():
            args += ["--predict_on_u_mask"]
        args = " ".join(args)
        Plugin.runSPFluo(self, Plugin.getProgram(PICKING_MODULE), args=args)

    def postprocessStep(self):
        args = [
            "--stages postprocess",
            f"--predictions {self._getExtraPath('picking', 'predictions.pickle')}",
            f"--checkpoint {os.path.abspath(self.trainRun._getExtraPath('picking', 'checkpoint.pt'))}",
            f"--testdir {self.test_dir}",
            f"--output_dir {self.output_dir}",
            f"--stride {self.stride.get()}",
            "--extension tif",
            "--iterative",
        ]
        if self.patchSize.get():
            args += (f"--patch_size {self.patchSize.get()}",)
        else:
            args += (
                f"--patch_size {self.trainRun.inputCoordinates.get().getBoxSize()}",
            )
        args = " ".join(args)
        Plugin.runSPFluo(self, Plugin.getProgram(PICKING_MODULE), args=args)

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
        coordSets: Dict[str, SetOfCoordinates3D] = {}
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
