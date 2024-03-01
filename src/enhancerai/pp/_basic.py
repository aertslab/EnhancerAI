from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from tqdm import tqdm

from enhancerai.utils import dependencies


def _dna_to_code(nt: str) -> int:
    if nt == "A":
        return 0
    elif nt == "C":
        return 1
    elif nt == "G":
        return 2
    elif nt == "T":
        return 3
    else:
        # scBasset does this
        return np.random.randint(0, 3)


@dependencies("genomepy")
def add_dna_sequence(
    adata: AnnData,
    seq_len: int = 2114,
    genome_name: str = "hg38",
    genome_dir: Path | None = None,
    genome_provider: str | None = None,
    install_genome: bool = True,
    chr_var_key: str = "chr",
    start_var_key: str = "start",
    end_var_key: str = "end",
    sequence_varm_key: str = "dna_sequence",
    code_varm_key: str = "dna_code",
) -> None:
    """Add DNA sequence to AnnData object.

    Uses genomepy under the hood to download the genome if no fasta_path provided.
    Code adapted from scvi-tools.preprocessing.


    Parameters
    ----------
    adata
        AnnData object with chromatin accessiblity data
    seq_len
        Length of DNA sequence to extract around peak center.
        Defaults to value used in ChromBPNet.
        You can still use a smaller value for `seq_len` in the model; consider this
        'seq_len' the maximum length you'll use, including padding/shift augmentations.
    genome_name
        Name of genome to use, installed with genomepy
    genome_dir
        Directory to install genome to, if not already installed
    genome_provider
        Provider of genome, passed to genomepy
    install_genome
        Install the genome with genomepy. If False, `genome_provider` is not used,
        and a genome is loaded with `genomepy.Genome(genome_name, genomes_dir=genome_dir)`
    chr_var_key
        Key in `.var` for chromosome
    start_var_key
        Key in `.var` for start position
    end_var_key
        Key in `.var` for end position
    sequence_varm_key
        Key in `.varm` for added DNA sequence
    code_varm_key
        Key in `.varm` for added DNA sequence, encoded as integers

    Returns
    -------
    None

    Adds fields to `.varm`:
        sequence_varm_key: DNA sequence
        code_varm_key: DNA sequence, encoded as integers
    """
    import genomepy

    if genome_dir is None:
        tempdir = tempfile.TemporaryDirectory()
        genome_dir = tempdir.name

    if install_genome:
        g = genomepy.install_genome(
            genome_name, genome_provider, genomes_dir=genome_dir
        )
    else:
        g = genomepy.Genome(genome_name, genomes_dir=genome_dir)

    chroms = adata.var[chr_var_key].unique()
    df = adata.var[[chr_var_key, start_var_key, end_var_key]]
    seq_dfs = []

    for chrom in tqdm(chroms):
        chrom_df = df[df[chr_var_key] == chrom]
        block_mid = (chrom_df[start_var_key] + chrom_df[end_var_key]) // 2
        block_starts = block_mid - (seq_len // 2)
        block_ends = block_starts + seq_len
        seqs = []

        for start, end in zip(block_starts, block_ends - 1):
            seq = str(g.get_seq(chrom, start, end)).upper()
            seqs.append(list(seq))

        assert len(seqs) == len(chrom_df)
        seq_dfs.append(pd.DataFrame(seqs, index=chrom_df.index))

    sequence_df = pd.concat(seq_dfs, axis=0).loc[adata.var_names]
    adata.varm[sequence_varm_key] = sequence_df
    adata.varm[code_varm_key] = sequence_df.applymap(_dna_to_code)


def train_val_test_split(
    adata: AnnData,
    test_size: float = 0.1,
    val_size: float = 0.1,
    shuffle: bool = True,
    random_state: None | int = None,
    type: str = "random",
    chr_val: list[str] = None,
    chr_test: list[str] = None,
    chr_var_key: str = "chr",
) -> None:
    """
    Add 'train/val/test' split column to AnnData object.

    This function adds a new column to the `.obs` or `.var` DataFrame of the AnnData object,
    indicating whether each sample should be part of the training, validation, or test set.

    Parameters
    ----------
    adata
        AnnData object to which the 'train/val/test' split column will be added.
    test_size
        Proportion of the dataset to include in the test split.
    val_size
        Proportion of the training dataset to include in the validation split.
    shuffle
        Whether or not to shuffle the data before splitting (when type='random').
    random_state
        When shuffle is True, random_state affects the ordering of the indices.
    type
        Type of split. Either 'random' or 'chr'. If 'chr', the "target" df should
        have a column "chr" with the chromosome names.
    chr_val
        List of chromosomes to include in the validation set. Required if type='chr'.
    chr_test
        List of chromosomes to include in the test set. Required if type='chr'.
    chr_var_key
        Key in `.var` for chromosome.

    Returns
    -------
    None

    Adds a new column to `adata.obs` or `adata.var`:
        'split': 'train', 'val', or 'test'
    """
    import math

    # Input checks
    if type not in ["random", "chr"]:
        raise ValueError("`type` should be either 'random' or 'chr'")
    if type == "chr":
        if chr_val is None or chr_test is None:
            raise ValueError(
                "If `type` is 'chr', `chr_val` and `chr_test` should be provided."
            )
        if chr_var_key not in adata.var.columns:
            raise ValueError(
                f"Column '{chr_var_key}' not found in `.var`. "
                "Make sure to add the chromosome information to the `.var` DataFrame."
            )
        unique_chr = adata.var[chr_var_key].unique()
        if not set(chr_val).issubset(unique_chr):
            raise ValueError(
                "Some chromosomes in `chr_val` are not present in the dataset."
            )
        if not set(chr_test).issubset(unique_chr):
            raise ValueError(
                "Some chromosomes in `chr_test` are not present in the dataset."
            )

    # Split
    n_samples = adata.n_vars

    if type == "random":
        if shuffle:
            np.random.seed(seed=random_state)
            indices = np.random.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        test_n = math.ceil(n_samples * test_size)
        val_n = math.ceil(n_samples * val_size)

        split = pd.Series("train", index=adata.var_names)
        split.iloc[indices[:test_n]] = "test"
        split.iloc[indices[test_n : test_n + val_n]] = "val"
    elif type == "chr":
        split = pd.Series("train", index=adata.var_names)
        split[adata.var[chr_var_key].isin(chr_test)] = "test"
        split[adata.var[chr_var_key].isin(chr_val)] = "val"

    adata.var["split"] = split
