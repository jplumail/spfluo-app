# -*- coding: utf-8 -*-
# **************************************************************************
# Module to declare viewers
# Find documentation here: https://scipion-em.github.io/docs/docs/developer/creating-a-viewer
# **************************************************************************

from singleparticle.viewers.imagej_viewers import ImageJViewer
from singleparticle.viewers.napari_viewers import NapariDataViewer

__all__ = [NapariDataViewer, ImageJViewer]
