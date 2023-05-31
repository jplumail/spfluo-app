from spfluo.objects.data import Coordinate3D, FluoImage, Particle, SetOfCoordinates3D, SetOfParticles
from spfluo.protocols.protocol_base import ProtFluoBase

from pyworkflow import BETA
from pyworkflow.protocol import Protocol, Form, params

import os
import numpy as np
from scipy.ndimage import affine_transform
from aicsimageio.aics_image import AICSImage


class ProtSPFluoExtractParticles(Protocol, ProtFluoBase):
    """Extract particles from a SetOfCoordinates"""
    OUTPUT_NAME = "SetOfParticles"
    _label = 'extract particles'
    _devStatus = BETA
    _possibleOutputs = {OUTPUT_NAME: SetOfParticles}

    def _defineParams(self, form: Form):
        form.addSection(label="Input")
        form.addParam(
            'inputCoordinates',
            params.PointerParam,
            pointerClass='SetOfCoordinates3D',
            label='coordinates you want to extract',
            important=True
        )
    
    def _insertAllSteps(self):
        self._insertFunctionStep(self.createOutputStep)
    
    def createOutputStep(self):
        particles = self._createSetOfParticles()
        coords: SetOfCoordinates3D = self.inputCoordinates.get()
        fluoimages = coords.getPrecedents()
        box_size = coords.getBoxSize()
        for im in fluoimages.iterItems():
            im: FluoImage
            for coord_im in coords.iterCoordinates(im):
                extracted_particle = self.extract_particle(im, coord_im, box_size)
                particles.append(extracted_particle)
        
        self._defineOutputs(**{self.OUTPUT_NAME: particles})
    
    def extract_particle(self, im: FluoImage, coord: Coordinate3D, box_size: int) -> Particle:
        mat = coord.getMatrix()
        mat[:3, 3] -= float(box_size) / 2
        image_data = im.getData()
        C = im.getNumChannels()
        particle_data = np.empty((1,C)+(box_size,)*3, dtype=image_data.dtype)
        for c in range(C):
            im_array_c = image_data[0, c] # T=0,C=c in AICS model
            particle_data[0, c] = affine_transform(im_array_c, mat, output_shape=(box_size,)*3)

        ext = os.path.splitext(im.getFileName())[1]
        coord_str = "-".join([f"{x:.2f}" for x in coord.getPosition()])
        name = im.getImgId() + '_' + coord_str + ext
        filepath = self._getExtraPath(name)
        new_particle = Particle(data=particle_data)
        new_particle.setImgId(os.path.basename(filepath))
        new_particle.setCoordinate3D(coord)
        new_particle.setImageName(im.getFileName())
        new_particle.setSamplingRate(im.getSamplingRate())
        # did not set origin, is it a problem?
        
        # save to disk
        new_particle.save(filepath)
        new_particle.setFileName(filepath)

        return new_particle