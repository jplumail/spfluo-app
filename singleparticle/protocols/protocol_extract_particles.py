from __future__ import annotations

import os

import numpy as np
from pwfluo.constants import MICRON_STR
from pwfluo.objects import (
    FluoImage,
    Particle,
    SetOfCoordinates3D,
    SetOfParticles,
)
from pwfluo.protocols import ProtFluoBase
from pyworkflow import BETA
from pyworkflow.protocol import Form, Protocol, params
from scipy.ndimage import affine_transform


def split_ome_tiff(filename):
    """
    Splits an OME-TIFF filename into the base name and extension.

    Args:
    filename (str): The OME-TIFF filename.

    Returns:
    tuple: A tuple containing the base name and extension.
    """
    if filename.endswith(".ome.tiff") or filename.endswith(".ome.tif"):
        base_name = filename.rsplit(".", 2)[0]
        extension = "." + filename.rsplit(".", 2)[1] + "." + filename.rsplit(".", 2)[2]
    else:
        # Fallback to regular splitext for non-OME-TIFF files
        base_name, extension = os.path.splitext(filename)

    return base_name, extension


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
            "size",
            params.FloatParam,
            label=f"Box size ({MICRON_STR}), optional",
            help="Size of the squared box surrounding the particle",
            expertLevel=params.LEVEL_ADVANCED,
            allowsNull=True,
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
        vs_xy, vs_z = fluoimages.getVoxelSize()
        if self.size.get() is not None:
            bs: float = self.size.get()
            box_size = (bs, bs, bs)
        else:
            box_size = None
        for im in fluoimages.iterItems():
            im: FluoImage
            image_data = im.getData()
            for coord_im in coords.iterCoordinates(im):
                particle_data = self.extract_particle(
                    image_data,
                    coord_im.getPosition(),
                    coord_im.getDim() if box_size is None else box_size,
                    voxel_size=im.getVoxelSize(),
                    subpixel=self.subpixel.get(),
                )

                ext = split_ome_tiff(im.getFileName())[1]
                coord_str = "-".join([f"{x:.2f}" for x in coord_im.getPosition()])
                name = im.getImgId() + "_" + coord_str + ext
                filepath = self._getExtraPath(name)

                extracted_particle = Particle.from_data(
                    particle_data,
                    filepath,
                    voxel_size=(vs_xy, vs_z),
                    num_channels=im.getNumChannels(),
                )
                extracted_particle.setCoordinate3D(coord_im)
                extracted_particle.setImageName(im.getFileName())
                extracted_particle.setImgId(os.path.basename(filepath))

                particles.append(extracted_particle)

        particles.write()

        self._defineOutputs(**{self.OUTPUT_NAME: particles})

    @staticmethod
    def extract_particle(
        image_data: np.ndarray,
        pos: tuple[float, float, float],
        box_size: tuple[float, float, float],
        voxel_size: tuple[float, float],
        subpixel: bool = False,
    ) -> np.ndarray:
        vs_xy, vs_z = voxel_size

        def world_to_data_coord(pos):
            return pos / np.asarray([vs_z, vs_xy, vs_xy])

        pos = world_to_data_coord(np.asarray(pos))  # World coordinates to data coords
        box_size_world = np.asarray(box_size, dtype=float)
        box_size_data = np.rint(world_to_data_coord(box_size_world)).astype(int)
        mat = np.eye(4)
        mat[:3, 3] = pos
        mat[:3, 3] -= box_size_data / 2
        C = image_data.shape[0]
        particle_data = np.empty((C,) + tuple(box_size_data), dtype=image_data.dtype)
        if not subpixel:
            top_left_corner = np.rint(mat[:3, 3]).astype(int)
            bottom_right_corner = top_left_corner + box_size_data
            xmin, ymin, zmin = top_left_corner
            xmax, ymax, zmax = bottom_right_corner
            original_shape = image_data.shape[1:]
            x_slice = slice(max(xmin, 0), min(xmax, original_shape[0]))
            y_slice = slice(max(ymin, 0), min(ymax, original_shape[1]))
            z_slice = slice(max(zmin, 0), min(zmax, original_shape[2]))
            x_overlap = slice(
                max(0, -xmin), min(box_size_data[0], original_shape[0] - xmin)
            )
            y_overlap = slice(
                max(0, -ymin), min(box_size_data[1], original_shape[1] - ymin)
            )
            z_overlap = slice(
                max(0, -zmin), min(box_size_data[2], original_shape[2] - zmin)
            )

        for c in range(C):
            if subpixel:
                particle_data[c] = affine_transform(
                    image_data[c], mat, output_shape=tuple(box_size_data)
                )
            else:
                padded_array = np.zeros(tuple(box_size_data), dtype=image_data.dtype)
                padded_array[x_overlap, y_overlap, z_overlap] = image_data[
                    c, x_slice, y_slice, z_slice
                ]
                particle_data[c] = padded_array.copy()

        return particle_data
