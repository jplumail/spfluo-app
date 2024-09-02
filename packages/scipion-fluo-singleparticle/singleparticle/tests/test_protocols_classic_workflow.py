from .test_fluo_base import TestProtocolFluoBase


class TestClassicWorkflow(TestProtocolFluoBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.runWorkflow()

    def test_workflow(self):
        self.assertIsNotNone(
            self.particles, "There was a problem with set of particles import"
        )
        self.assertIsNotNone(self.psf, "There was a problem with psf import")
        self.assertIsNotNone(self.output_vol, "Problem with reconstructed volume")
        self.assertIsNotNone(self.output_particles, "Problem with output particles")
        self.assertIsNotNone(self.aligned_vol, "Problem with aligned volume")
        self.assertIsNotNone(self.aligned_particles, "Problem with aligned particles")
        self.assertIsNotNone(self.refinement_vol, "Problem with refinement_vol")
        self.assertIsNotNone(
            self.refinement_particles, "Problem with refinement particles"
        )

    def test_abinitio_gpu(self):
        self.runAbinitio(gpu=True)

    def test_abinitio_nogpu(self):
        self.runAbinitio(gpu=False)

    def test_abinitio_minibatch(self):
        self.runAbinitio(minibatch=0)

    def test_abinitio_pad(self):
        self.runAbinitio(pad=True, minibatch=128)

    def test_abinitio_num_iter(self):
        self.runAbinitio(numIterMax=2)
