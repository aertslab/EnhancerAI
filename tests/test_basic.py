import pytest
from anndata import AnnData

import enhancerai as enhai


def test_package_has_version():
    assert enhai.__version__ is not None


def test_import_topics_returns_anndata():
    ann_data = enhai.import_topics(
        topics_folder="tests/data/test_topics", peaks_file="tests/data/test.peaks.bed"
    )
    assert isinstance(ann_data, AnnData)


def test_import_topics_correct_shape():
    ann_data = enhai.import_topics(
        topics_folder="tests/data/test_topics", peaks_file="tests/data/test.peaks.bed"
    )
    expected_number_of_topics = 3
    expected_number_of_peaks = 23186

    assert ann_data.shape == (expected_number_of_topics, expected_number_of_peaks)


def test_import_topics_obs_vars_dataframe():
    ann_data = enhai.import_topics(
        topics_folder="tests/data/test_topics", peaks_file="tests/data/test.peaks.bed"
    )
    assert "file_path" in ann_data.obs.columns
    assert "n_open_regions" in ann_data.obs.columns
    assert "n_topics" in ann_data.var.columns


def test_import_topics_topics_subset():
    ann_data = enhai.import_topics(
        topics_folder="tests/data/test_topics",
        peaks_file="tests/data/test.peaks.bed",
        topics_subset=["Topic_1", "Topic_2"],
    )
    assert ann_data.shape[0] == 2


def test_import_topics_invalid_files():
    with pytest.raises(FileNotFoundError):
        enhai.import_topics(topics_folder="invalid_folder", peaks_file="invalid_file")
