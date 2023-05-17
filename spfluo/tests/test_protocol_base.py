from pyworkflow.tests import BaseTest, setupTestProject, DataSet

from spfluo.protocols.protocol_import import ProtImportFluoImages


class TestProtocolBase(BaseTest):
    ds = None
    sr_xy = 1.1
    sr_z = 6.32

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet("fluo")
    
    @classmethod
    def runImportFluoImages(cls):
        protImportFluoImages = cls.newProtocol(
            ProtImportFluoImages,
            filesPath=cls.ds.getFile("fluo1"),
            samplingRate=(cls.sr_xy, cls.sr_z),
        )

        cls.launchProtocol(protImportFluoImages)
        fluoimagesImported = protImportFluoImages.FluoImages
        print(fluoimagesImported)
        cls.assertIsNotNone(fluoimagesImported, "There was a problem with tomogram output")