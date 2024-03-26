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
        loss=nn.BCELoss(),
        optimizer=torch.optim.Adam,
        lr=1e-3,
        **optimizer_kwargs,
    ):
        super().__init__()
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs
        self.model = None

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

    def initialize_model(self, model):
        self.model = model

    def forward(self, x):
        if self.model is None:
            raise ValueError("Model architecture not initialized")
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(
            self.train_metrics(y_hat, y), prog_bar=False, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(
            self.val_metrics(y_hat, y), prog_bar=False, on_step=False, on_epoch=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("test/loss", loss, on_step=False, prog_bar=False)
        self.log_dict(self.test_metrics(y_hat, y), on_step=False, prog_bar=False)
        return loss

    def predict_step(self, batch):
        x, _ = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        return self.optimizer(
            self.model.parameters(), lr=self.lr, **self.optimizer_kwargs
        )

    def get_params(self):
        return {
            "task": self.__class__.__name__,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "optimizer_kwargs": self.optimizer_kwargs,
        }
