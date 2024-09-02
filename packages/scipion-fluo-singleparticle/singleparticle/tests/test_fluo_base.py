from pwfluo.objects import AverageParticle, SetOfParticles
from pwfluo.protocols import ProtImportPSFModel, ProtImportSetOfParticles
from pyworkflow.tests import BaseTest, DataSet, setupTestProject

from singleparticle.protocols import (
    ProtSingleParticleAbInitio,
    ProtSingleParticleAlignAxis,
    ProtSingleParticleRefinement,
)


class TestProtocolFluoBase(BaseTest):
    ds = None

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.ds: DataSet = DataSet.getDataSet("fluo")

    @classmethod
    def runWorkflow(cls):
        cls.particles = cls.runImportSetOfParticles()
        cls.psf = cls.runImportPSF()
        cls.output_vol, cls.output_particles = cls.runAbinitio()
        cls.aligned_vol, cls.aligned_particles = cls.runAlignAxis()
        cls.refinement_vol, cls.refinement_particles = cls.runRefinement()

    @classmethod
    def runImportSetOfParticles(cls):
        prot = cls.newProtocol(
            ProtImportSetOfParticles,
            filesPath=cls.ds.getFile("isotropic-particles-dir") + "/*",
            vs_xy=1.0,
            vs_z=1.0,
        )

        cls.launchProtocol(prot)
        output = prot.SetOfParticles
        return output

    @classmethod
    def runImportPSF(cls):
        protImportPSFModel = cls.newProtocol(
            ProtImportPSFModel,
            filePath=cls.ds.getFile("isotropic-psf"),
            vs_xy=1.0,
            vs_z=1.0,
        )

        cls.launchProtocol(protImportPSFModel)
        psfImported = protImportPSFModel.PSFModel
        return psfImported

    @classmethod
    def runAbinitio(cls, **kwargs):
        default_kwargs = {
            "pad": False,
            "minibatch": 256,
            "gpu": True,
            "channel": 0,
            "numIterMax": 1,
        }
        default_kwargs.update(kwargs)
        prot = cls.newProtocol(
            ProtSingleParticleAbInitio,
            inputParticles=cls.particles,
            inputPSF=cls.psf,
            **default_kwargs
        )
        cls.launchProtocol(prot)
        output_vol: AverageParticle = prot.reconstructedVolume
        output_particles: SetOfParticles = prot.particles
        return output_vol, output_particles

    @classmethod
    def runAlignAxis(cls):
        prot = cls.newProtocol(
            ProtSingleParticleAlignAxis,
            inputParticle=cls.output_vol,
            inputParticles=cls.output_particles,
            sym=9,
        )
        cls.launchProtocol(prot)
        aligned_vol = prot.aligned_volume
        aligned_particles = prot.aligned_particles
        return aligned_vol, aligned_particles

    @classmethod
    def runRefinement(cls, **kwargs):
        default_kwargs = {
            "pad": False,
            "N_axes": 2,
            "N_rot": 2,
            "sym": 9,
            "minibatch": 128,
        }
        default_kwargs.update(kwargs)
        prot = cls.newProtocol(
            ProtSingleParticleRefinement,
            inputParticles=cls.aligned_particles,
            inputPSF=cls.psf,
            **default_kwargs
        )
        cls.launchProtocol(prot)
        refinement_vol = prot.reconstructedVolume
        particles = prot.particles
        return refinement_vol, particles
