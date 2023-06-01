from pyworkflow.tests import BaseTest, setupTestProject, DataSet

from spfluo.protocols.protocol_import import ProtImportFluoImages


class TestProtocolBase(BaseTest):
    ds = None
    vs_xy = 1.1
    vs_z = 6.32

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet("fluo")
    
    @classmethod
    def runImportFluoImages(cls):
        protImportFluoImages = cls.newProtocol(
            ProtImportFluoImages,
            filesPath=cls.ds.getFile("fluo1"),
            voxelSize=(cls.vs_xy, cls.vs_z),
        )

        cls.launchProtocol(protImportFluoImages)
        fluoimagesImported = protImportFluoImages.FluoImages
        print(fluoimagesImported)
        cls.assertIsNotNone(fluoimagesImported, "There was a problem with tomogram output")