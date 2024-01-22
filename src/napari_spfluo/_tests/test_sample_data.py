from napari_spfluo import make_generated_anisotropic


def test_aniso():
    assert len(make_generated_anisotropic()) == 3
