from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class DeepTopic(L.LightningModule):
    def __init__(
        self,
        model,
        loss=nn.BCELoss(),
        optimizer=torch.optim.Adam,
        lr=1e-3,
        **optimizer_kwargs,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs

        # Metrics
        metrics = MetricCollection(
            [
                BinaryAccuracy(),
                BinaryAUROC(),
                BinaryF1Score(),
                BinaryPrecision(),
                BinaryRecall(),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.save_hyperparameters(
            ignore=["model", "loss", "train_metrics", "val_metrics", "test_metrics"]
        )

    def forward(self, x, y):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(
            self.train_metrics(y_hat, y), prog_bar=False, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(
            self.val_metrics(y_hat, y), prog_bar=False, on_step=False, on_epoch=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(
            batch,
            batch_idx,
        )
        return loss

    def configure_optimizers(self):
        return self.optimizer(
            self.model.parameters(), lr=self.lr, **self.optimizer_kwargs
        )
