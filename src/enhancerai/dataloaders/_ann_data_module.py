from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from scipy.sparse import spmatrix
from torch.utils.data import DataLoader, Dataset


class AnnDataSet(Dataset):
    def __init__(
        self,
        anndata,
        indices,
        code_varm_key: str = "dna_code",
        in_memory: bool = False,
    ):
        self.anndata = anndata
        self.indices = indices
        self.code_varm_key = code_varm_key
        self.in_memory = in_memory
        self.compressed = isinstance(self.anndata.X, spmatrix)

        if self.in_memory:
            seq_ids = torch.from_numpy(
                self.anndata.varm[self.code_varm_key].loc[self.indices].values
            )
            self.one_hot = F.one_hot(seq_ids, num_classes=4)
            self.one_hot = self.one_hot.permute(0, 2, 1).to(
                torch.float32
            )  # (N, 4, seq_len)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]

        if self.in_memory:
            x = self.one_hot[idx, :, :]
        else:
            seq_id = torch.from_numpy(
                self.anndata.varm[self.code_varm_key].loc[index].values
            )
            x = F.one_hot(seq_id, num_classes=4)
            x = x.permute(1, 0).to(torch.float32)  # (4, seq_len)

        if self.compressed:
            y = torch.tensor(
                self.anndata.X[:, idx].todense(), dtype=torch.float32
            ).squeeze(1)
        else:
            y = torch.tensor(self.anndata.X[:, idx], dtype=torch.float32).squeeze(1)

        return x, y


class AnnDataModule(L.LightningDataModule):
    def __init__(
        self,
        anndata,
        batch_size: int,
        shuffle: bool = True,
        in_memory: bool = False,
        num_workers: int = 0,
        code_varm_key: str = "dna_code",
    ):
        super().__init__()
        self.anndata = anndata
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.in_memory = in_memory
        self.num_workers = num_workers
        self.code_varm_key = code_varm_key

        if "split" not in self.anndata.var.columns:
            raise ValueError(
                "The AnnData object should have a 'split' column in `.var`. Run `pp.train_val_test_split` first."
            )
        if code_varm_key not in self.anndata.varm:
            raise ValueError(
                f"The AnnData object should have a '{code_varm_key}' column in `.varm`. Run `pp.add_dna_sequence` first."
            )

    def setup(self, stage=None):
        """Get train/val/test indices from the AnnData object."""
        self.train_idx = self.anndata.var_names[self.anndata.var["split"] == "train"]
        self.val_idx = self.anndata.var_names[self.anndata.var["split"] == "val"]
        self.test_idx = self.anndata.var_names[self.anndata.var["split"] == "test"]

    def train_dataloader(self):
        return DataLoader(
            AnnDataSet(
                self.anndata, self.train_idx, self.code_varm_key, self.in_memory
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            AnnDataSet(self.anndata, self.val_idx, self.code_varm_key, self.in_memory),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            AnnDataSet(self.anndata, self.test_idx, self.code_varm_key, self.in_memory),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


if __name__ == "__main__":
    # Testing
    # TODO:remove
    import anndata as ad

    adata = ad.read_h5ad(
        "/home/luna.kuleuven.be/u0166574/Desktop/projects/EnhancerAI/test.h5ad"
    )
    mod = AnnDataModule(adata, batch_size=32)
    mod.setup()

    for x, y in mod.train_dataloader():
        print(x.shape, y.shape)
        break

    import time

    start = time.time()
    for _, _ in mod.train_dataloader():
        pass
    print("not in memory", time.time() - start)

    mod = AnnDataModule(adata, batch_size=32, in_memory=True)
    mod.setup()

    for x, y in mod.train_dataloader():
        print(x.shape, y.shape)
        break

    start = time.time()
    for _, _ in mod.train_dataloader():
        pass
    print("in memory", time.time() - start)
