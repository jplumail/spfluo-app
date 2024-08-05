from __future__ import annotations

from pwfluo.objects import FluoSet, SetOfParticles
from pwfluo.protocols import ProtFluoBase
from pyworkflow import BETA
from pyworkflow.object import RELATION_SOURCE
from pyworkflow.protocol import Form, Protocol, params

from singleparticle.viewers.napari_viewers import SetOfParticlesView


class ProtSingleParticleSelectSubset(Protocol, ProtFluoBase):
    """
    Select a subset of a set
    """

    OUTPUT_PREFIX = "Selected"
    _label = "select subset"
    _devStatus = BETA
    _possibleOutputs = FluoSet

    # -------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form: Form):
        form.addSection(label="Data params")
        form.addParam(
            "inputSet",
            params.PointerParam,
            pointerClass="FluoSet",
            label="Set",
            important=True,
        )

    def _insertAllSteps(self):
        self._insertFunctionStep(self.selectSubsetStep)

    def selectSubsetStep(self):
        self.inputSet: FluoSet = self.inputSet.get()
        if isinstance(self.inputSet, SetOfParticles):
            self.outputSet = self._createSetOfParticles(
                self._getOutputSuffix(SetOfParticles)
            )
            view = SetOfParticlesView(self.inputSet)

            w = view.show()
            w.process.join()
            list_indices = w.queue.get()
            self.createOutputSetOfParticles(list_indices)

    def createOutputSetOfParticles(self, indices: list[int]):
        self.outputSet.enableAppend()
        for j, particle in enumerate(self.inputSet.iterItems()):
            print(j, particle.getFileName(), end="")
            if j not in indices:
                print(" NOT KEPT")
                continue
            print(" KEPT")
            self.outputSet.append(particle)
        self.outputSet.write()
        name = self.OUTPUT_PREFIX + self._getOutputSuffix(SetOfParticles)
        self._defineOutputs(**{name: self.outputSet})
        self._defineRelation(RELATION_SOURCE, self.inputSet, self.outputSet)
