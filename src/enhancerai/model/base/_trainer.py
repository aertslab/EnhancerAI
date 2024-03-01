from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger


class Trainer:
    def __init__(
        self,
        model,
        datamodule,
        project_name: str,
        experiment_name: str | None = None,
        logger: str | None = "wandb",
        model_checkpointing: bool = True,
        model_checkpointing_dir: str = "checkpoints",
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        early_stopping_mode: str = "min",
        early_stopping_monitor: str = "val/loss",
        callbacks: list | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        self.model = model
        self.datamodule = datamodule
        self.seed = seed
        self.kwargs = kwargs

        # callbacks
        if early_stopping:
            early_stopping_callback = EarlyStopping(
                patience=early_stopping_patience,
                mode=early_stopping_mode,
                monitor=early_stopping_monitor,
                verbose=True,
            )
        else:
            early_stopping_callback = None

        if model_checkpointing:
            model_checkpoint_callback = ModelCheckpoint(
                dirpath=model_checkpointing_dir,
                filename="best_model",
                monitor=early_stopping_monitor,
                mode=early_stopping_mode,
                save_top_k=1,
                verbose=True,
            )
        else:
            model_checkpoint_callback = None

        learning_rate_monitor_callback = LearningRateMonitor(logging_interval="step")

        self.callbacks = [
            early_stopping_callback,
            model_checkpoint_callback,
            learning_rate_monitor_callback,
        ]
        if callbacks is not None:
            self.callbacks.extend(callbacks)

        # logger
        if logger is not None:
            if logger == "wandb":
                self.logger = WandbLogger(
                    name=experiment_name, project=project_name, log_model=False
                )
            elif logger == "tensorboard":
                self.logger = TensorBoardLogger(
                    save_dir=project_name,
                    name=experiment_name,
                )
            else:
                raise ValueError("Invalid logger, use 'None', 'wandb' or 'tensorboard'")
        else:
            self.logger = None

    def fit(self):
        trainer = L.Trainer(callbacks=self.callbacks, logger=self.logger, **self.kwargs)
        trainer.fit(self.model, self.datamodule)

    def test(self):
        trainer = L.Trainer(**self.kwargs)
        trainer.test(self.model, self.datamodule)

    def predict(self):
        trainer = L.Trainer(**self.kwargs)
        trainer.predict(self.model, self.datamodule)
