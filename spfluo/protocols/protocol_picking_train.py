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
import random
from pyworkflow import BETA
from pyworkflow.protocol import Protocol, params, Integer
from pyworkflow.utils import Message
import tomo.constants

from spfluo import Plugin
from spfluo.constants import *
from spfluo.convert import convert_to_tif, write_csv


class ProtSPFluoPickingTrain(Protocol):
    """
    Picking for fluo data with deep learning
    """
    _label = 'picking train'
    _devStatus = BETA

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label="Data params")
        group = form.addGroup("Input")
        group.addParam(
            'inputCoordinates', params.PointerParam, pointerClass='SetOfCoordinates3D',
            label='Annotations 3D coordinates', important=True
        )
        form.addParam(
            'pu',
            params.BooleanParam,
            label='Positive Unlabelled learning',
            default=True,
            expertLevel=params.LEVEL_ADVANCED
        )
        group = form.addGroup("PU params", condition='pu')
        group.addParam(
            'num_particles_per_image',
            params.IntParam,
            default=None,
            condition='pu',
            label='Number of particles per image',
        )
        group.addParam(
            'radius',
            params.IntParam,
            default=None,
            condition='pu',
            label='Radius',
            expertLevel=params.LEVEL_ADVANCED,
            allowsNull=True
        )
        form.addSection(label="Advanced", expertLevel=params.LEVEL_ADVANCED)
        form.addParam(
            'lr',
            params.FloatParam,
            label='Learning rate',
            default=1e-3,
        )
        group = form.addGroup("Data params")
        group.addParam(
            'train_val_split',
            params.FloatParam,
            default=0.7,
            label="Train/val split",
            help="By default 70% of the data is in the training set",
        )
        group.addParam(
            'batch_size',
            params.IntParam,
            label='Batch size',
            default=128,
        )
        group.addParam(
            'epoch_size',
            params.IntParam,
            label='epoch size',
            default=20,
        )
        group.addParam(
            'num_epochs',
            params.IntParam,
            label='num epochs',
            default=5,
        )
        group.addParam(
            'shuffle',
            params.BooleanParam,
            label='Shuffle samples at each epoch',
            default=True,
        )
        group.addParam(
            'augment',
            params.FloatParam,
            label='Augment rate',
            default=0.8,
        )
        # SWA
        form.addParam(
            'swa',
            params.BooleanParam,
            label='Enable SWA',
            default=True,
            help='Stochastic Weight Averaging',
            expertLevel=params.LEVEL_ADVANCED
        )
        group = form.addGroup("SWA params", condition='swa')
        group.addParam(
            'swa_lr',
            params.FloatParam,
            condition='swa',
            label='SWA learning rate',
            default=1e-5,
            expertLevel=params.LEVEL_ADVANCED
        )
    
    # --------------------------- STEPS functions ------------------------------
    def _insertAllSteps(self):
        self.pickingPath = os.path.abspath(self._getExtraPath(PICKING_WORKING_DIR))
        self.rootDir = os.path.join(self.pickingPath, "rootdir")
        self._insertFunctionStep(self.prepareStep)
        self._insertFunctionStep(self.trainStep)
    
    def prepareStep(self):
        if not os.path.exists(self.rootDir):
            os.makedirs(self.rootDir, exist_ok=True)
        os.makedirs(os.path.join(self.rootDir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.rootDir, 'val'), exist_ok=True)

        # Image links
        images = set([coord.getVolume() for coord in self.inputCoordinates.get().iterCoordinates()])
        for im in images:
            im_path = os.path.abspath(im.getFileName())
            ext = os.path.splitext(im_path)[1]
            im_name = im.getTsId()
            im_newPath = os.path.join(self.rootDir, im_name+'.tif')
            if ext != '.tif' and ext != '.tiff':
                print(f"Convert {im_path} to TIF in {im_newPath}")
                convert_to_tif(im_path, im_newPath)
            else:
                os.link(im_path, im_newPath)
            for s in ["train", "val"]:
                im_newPathSet = os.path.join(self.rootDir, s, im_name+'.tif')
                if not os.path.exists(im_newPathSet):
                    print(f"Link {im_newPath} -> {im_newPathSet}")
                    os.link(im_newPath, im_newPathSet)
        
        # Splitting annotations in train/val
        origin_func = tomo.constants.SCIPION
        
        annotations = []
        for i, coord in enumerate(self.inputCoordinates.get().iterCoordinates()):
            Lx, Ly, Lz = coord.getVolume().getDim()
            annotations.append((
                coord.getVolume().getTsId()+'.tif',
                i,
                -(coord.getZ(origin_func)-Lz/2),
                coord.getY(origin_func)+Ly/2,
                coord.getX(origin_func)+Lx/2
            ))

        print(f"Found {len(annotations)} annotations in SetOfCoordinates created at {self.inputCoordinates.get().getObjCreationAsDate()}")
        random.shuffle(annotations)
        i = int(self.train_val_split.get() * len(annotations))
        train_annotations, val_annotations = annotations[:i], annotations[i:]

        # Write CSV
        write_csv(os.path.join(self.rootDir, 'train', 'train_coordinates.csv'), train_annotations)
        write_csv(os.path.join(self.rootDir, 'val', 'val_coordinates.csv'), val_annotations)

    def trainStep(self):
        ps = self.inputCoordinates.get().getBoxSize()
        args = [
            f"--stages train",
            f"--batch_size {self.batch_size.get()}",
            f"--rootdir {self.rootDir}",
            f"--output_dir {self.pickingPath}",
            f"--patch_size {ps}",
            f"--epoch_size {self.epoch_size.get()}",
            f"--num_epochs {self.num_epochs.get()}",
            f"--lr {self.lr.get()}",
            f"--extension tif",
            f"--augment {self.augment.get()}"
        ]
        if self.pu:
            args += [f"--mode pu"]
            if self.radius.get() is None:
                args += [f"--radius {ps//2}"]
            else:
                args += [f"--radius {self.radius.get()}"]
            args += [f"--num_particles_per_image {self.num_particles_per_image.get()}"]
        else:
            args += ["--mode fs"]
        if self.shuffle.get():
            args += ["--shuffle"]
        if self.swa.get():
            args += ["--swa", f"--swa_lr {self.swa_lr.get()}"]
        args = " ".join(args)
        Plugin.runSPFluo(self, Plugin.getProgram(PICKING_MODULE), args=args)
    
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
