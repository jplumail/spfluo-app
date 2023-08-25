import os

import numpy as np
from pwfluo.objects import (
    Coordinate3D,
    FluoImage,
    Particle,
    SetOfCoordinates3D,
    SetOfParticles,
)
from pwfluo.protocols import ProtFluoBase
from pyworkflow import BETA
from pyworkflow.protocol import Form, Protocol, params
from scipy.ndimage import affine_transform


class ProtSingleParticleExtractParticles(Protocol, ProtFluoBase):
    """Extract particles from a SetOfCoordinates"""

    OUTPUT_NAME = "SetOfParticles"
    _label = "extract particles"
    _devStatus = BETA
    _possibleOutputs = {OUTPUT_NAME: SetOfParticles}

    def _defineParams(self, form: Form):
        form.addSection(label="Input")
        form.addParam(
            "inputCoordinates",
            params.PointerParam,
            pointerClass="SetOfCoordinates3D",
            label="coordinates you want to extract",
            important=True,
        )
        form.addParam(
            "subpixel",
            params.BooleanParam,
            label="Subpixel precision?",
            default=False,
            expertLevel=params.LEVEL_ADVANCED,
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
                extracted_particle = self.extract_particle(
                    im, coord_im, box_size, subpixel=self.subpixel.get()
                )

                ext = os.path.splitext(im.getFileName())[1]
                coord_str = "-".join([f"{x:.2f}" for x in coord_im.getPosition()])
                name = im.getImgId() + "_" + coord_str + ext
                filepath = self._getExtraPath(name)
                extracted_particle.setImgId(os.path.basename(filepath))

                # save to disk
                extracted_particle.save(filepath)
                extracted_particle.setFileName(filepath)

                particles.append(extracted_particle)

        particles.write()

        self._defineOutputs(**{self.OUTPUT_NAME: particles})

    @staticmethod
    def extract_particle(
        im: FluoImage, coord: Coordinate3D, box_size: int, subpixel: bool = False
    ) -> Particle:
        vs_xy, vs_z = im.getVoxelSize()
        world_to_data_coord = lambda pos: pos / np.asarray([vs_z, vs_xy, vs_xy])
        mat = coord.getMatrix()
        mat[:3, 3] = world_to_data_coord(mat[:3, 3]) # World coordinates to data coords
        box_size_world = np.asarray([box_size, box_size, box_size], dtype=float)
        box_size_data = np.rint(world_to_data_coord(box_size_world)).astype(int)
        mat[:3, 3] -= box_size_data / 2
        mat[:3, :3] = np.eye(3)
        image_data = im.getData()
        C = im.getNumChannels()
        particle_data = np.empty((1, C) + tuple(box_size_data), dtype=image_data.dtype)
        if not subpixel:
            top_left_corner = np.rint(mat[:3, 3]).astype(int)
            bottom_right_corner = top_left_corner + box_size_data
            xmin, ymin, zmin = top_left_corner
            xmax, ymax, zmax = bottom_right_corner
        for c in range(C):
            im_array_c = image_data[0, c]  # T=0,C=c in AICS model
            if subpixel:
                particle_data[0, c] = affine_transform(
                    im_array_c, mat, output_shape=tuple(box_size_data)
                )
            else:
                particle_data[0, c] = im_array_c[xmin:xmax, ymin:ymax, zmin:zmax]

        new_particle = Particle(data=particle_data)
        new_particle.setCoordinate3D(coord)
        new_particle.setImageName(im.getFileName())
        new_particle.setVoxelSize(im.getVoxelSize())
        # did not set origin, is it a problem?
        return new_particle
