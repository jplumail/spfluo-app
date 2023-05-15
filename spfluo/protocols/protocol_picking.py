import os
from spfluo import Plugin

from pyworkflow import BETA

from spfluo.napari.multiple_viewer_widget import annotate_ortho_view
from .protocol_base import ProtFluoPicking
from pyworkflow.protocol import Protocol, params, Form


class ProtSPFluoPickingNapari(ProtFluoPicking):
    """
    Picking with the Napari plugin.
    """
    _label = 'manual picking'
    _devStatus = BETA

    def __init__(self, **kwargs):
        ProtFluoPicking.__init__(self, **kwargs)
        self.script_path = os.path.join(
            os.path.dirname(__file__), "..", "napari", "multiple_viewer_widget.py"
        )
    
    def _defineParams(self, form: Form):
        ProtFluoPicking._defineParams(self, form)
    
    def _insertAllSteps(self):
        self._insertFunctionStep(self.pickingStep)
    
    def pickingStep(self):
        im = self.inputFluoImages.get() # TODO
        annotate_ortho_view(im)