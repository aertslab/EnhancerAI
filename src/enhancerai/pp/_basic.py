from __future__ import annotations

from pathlib import Path

from anndata import AnnData


def import_topics(topics_folder: Path, peaks_file: Path, topics_subset: list | None = None) -> AnnData:
    """
    Import topic and consensus regions BED files into AnnData format.

    This format is required to be able to train a DeepTopic model.
    The topic and consensus regions are the outputs from running pycisTopic
    (https://pycistopic.readthedocs.io/en/latest/) on your data.
    The result is an AnnData object with topics as rows and peaks as columns,
    with binary values indicating whether a peak is present in a topic.

    Parameters
    ----------
    topics_folder
        Folder name containing the topic BED files.
    peaks_file
        File name of the consensus regions BED file.
    topics_subset
        List of topics to include in the AnnData object. If None, all topics
        will be included.

    Returns
    -------
    AnnData object with topics as rows and peaks as columns.
    """
    # Generate a random AnnData object for now
    adata = AnnData()

    return adata
