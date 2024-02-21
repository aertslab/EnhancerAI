import enhancerai as enhai


def test_package_has_version():
    assert enhai.__version__ is not None


def test_import_topics():
    topics_folder = "tests/data/test_topics/"
    peaks_file = "tests/data/test.peaks.bed"
    adata = enhai.pp.import_topics(topics_folder, peaks_file)

    assert adata is not None
