import numpy as np
import pytest
from anndata import AnnData

from enhancerai.pp import train_val_test_split


def test_train_val_test_split():
    # Create a dummy AnnData object
    adata = AnnData(np.random.rand(100, 100))
    adata.var["chr"] = ["chr1"] * 50 + ["chr2"] * 50

    train_val_test_split(
        adata, test_size=0.1, val_size=0.1, shuffle=True, random_state=42
    )

    # Check that the 'split' column was added to adata.var
    assert "split" in adata.var.columns

    # Check that the 'split' column contains the correct categories
    assert set(adata.var["split"]) == {"train", "val", "test"}

    # check that the number of samples in each category is correct
    n_samples = adata.n_vars
    n_train = sum(adata.var["split"] == "train")
    n_val = sum(adata.var["split"] == "val")
    n_test = sum(adata.var["split"] == "test")

    assert n_train + n_val + n_test == n_samples
    assert n_train > 0
    assert n_val > 0
    assert n_test > 0
    assert n_test / n_samples == 0.1
    assert n_val / n_samples == 0.1

    # Call the function with type='chr', chr_val=['chr1'], and chr_test=['chr2']
    train_val_test_split(adata, type="chr", chr_val=["chr1"], chr_test=["chr2"])
    assert "split" in adata.var.columns
    assert set(adata.var.loc[adata.var["chr"] == "chr1", "split"]) == {"val"}
    assert set(adata.var.loc[adata.var["chr"] == "chr2", "split"]) == {"test"}

    # Call the function with an invalid type
    with pytest.raises(ValueError):
        train_val_test_split(adata, type="invalid")

    # Call the function with type='chr' but without chr_val and chr_test
    with pytest.raises(ValueError):
        train_val_test_split(adata, type="chr")

    # Call the function with type='chr' and an invalid chr_var_key
    with pytest.raises(ValueError):
        train_val_test_split(
            adata,
            type="chr",
            chr_val=["chr1"],
            chr_test=["chr2"],
            chr_var_key="invalid",
        )

    # Call the function with type='chr' and an invalid chr_val
    with pytest.raises(ValueError):
        train_val_test_split(
            adata,
            type="chr",
            chr_val=["chr31"],
            chr_test=["chr2"],
        )
