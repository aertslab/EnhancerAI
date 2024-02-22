import tempfile
from pathlib import Path
from typing import Optional

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
    genome_dir: Optional[Path] = None,
    genome_provider: Optional[str] = None,
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
