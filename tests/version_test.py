import larq_zoo


def test_version():
    assert hasattr(larq_zoo, "__version__") and "." in larq_zoo.__version__
