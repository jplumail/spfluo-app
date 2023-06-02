from pwem.viewers.viewer_volumes import viewerProtImportVolumes  # FIXME
from ..protocols import ProtSPFluoAbInitio


class ReconstructViewer(viewerProtImportVolumes):
    _label = "viewer reconstruction"
    _targets = [ProtSPFluoAbInitio]
