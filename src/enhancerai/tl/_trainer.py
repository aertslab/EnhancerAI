from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

from enhancerai.utils import dependencies


class Trainer:
    def __init__(
        self,
        project_name: str,
        experiment_name: str | None = None,
        logger_type: str | None = "wandb",
        model_checkpointing: bool = True,
        model_checkpointing_dir: str = "checkpoints",
        early_stopping: bool = True,
        early_stopping_patience: int = 10,
        early_stopping_mode: str = "min",
        early_stopping_monitor: str = "val/loss",
        custom_callbacks: list | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        self.seed = seed
        self.kwargs = kwargs
        self.logger_type = logger_type
        self.project_name = project_name
        self.experiment_name = experiment_name

        self._initialize_callbacks(
            early_stopping,
            early_stopping_patience,
            early_stopping_mode,
            early_stopping_monitor,
            model_checkpointing,
            model_checkpointing_dir,
            custom_callbacks,
        )

    def fit(self):
        trainer = L.Trainer(callbacks=self.callbacks, logger=self.logger, **self.kwargs)
        trainer.fit(self.task, self.datamodule)

    def test(self):
        trainer = L.Trainer(**self.kwargs)
        trainer.test(self.task, self.datamodule)

    def predict(self):
        trainer = L.Trainer(**self.kwargs)
        return trainer.predict(self.task, self.datamodule)

    def setup(self, model, task, datamodule, experiment_name: str | None = None):
        self.model_architecture = model
        self.task = task
        self.datamodule = datamodule
        self.task.initialize_model(self.model_architecture)

        self._initialize_logger(
            self.logger_type, self.project_name, self.experiment_name
        )

    def sweep(
        self,
        sweep_config: dict | None = None,
    ):
        import wandb

        # You might want to modify this part if you have specific initialization for different architectures.
        sweep_id = wandb.sweep(sweep_config, project=self.project_name)
        wandb.agent(
            sweep_id, function=self._sweep_fit, count=sweep_config.get("max_trials", 1)
        )
        return sweep_id

    def _sweep_fit(self):
        import wandb

        config = wandb.config

        # Dynamic architecture instantiation based on sweep parameters
        architecture_class = self._get_architecture_class(
            config.architecture.architecture
        )
        architecture = architecture_class(**config.architecture)

        # Initialize and fit model (you may need to adjust this part according to your setup)
        self.setup(architecture, self.task, self.datamodule)
        self.fit()

    def _initialize_logger(self, logger, project_name, experiment_name):
        @dependencies("wandb")
        def _wandb_logger(experiment_name, project_name):
            self.logger = WandbLogger(
                name=experiment_name, project=project_name, log_model=False
            )
            self.logger.watch(self.task, log="gradients", log_graph=True)
            model_params = self.model_architecture.get_params()
            task_params = self.task.get_params()
            data_params = self.datamodule.get_params()
            self.logger.experiment.config.update(
                {
                    "architecture": model_params,
                    "task": task_params,
                    "datamodule": data_params,
                }
            )

        if logger is not None:
            if logger == "wandb":
                _wandb_logger(experiment_name, project_name)
                self.project_name = project_name
            else:
                raise ValueError("Invalid logger, use 'None' or 'wandb'")
        else:
            self.logger = None

    def _get_architecture_class(self, architecture_type):
        if architecture_type == "deeptopic":
            from enhancerai.tl.zoo import DeepTopicCNN

            return DeepTopicCNN
        else:
            raise ValueError(f"Architecture type {architecture_type} not found")

    def _initialize_callbacks(
        self,
        early_stopping,
        early_stopping_patience,
        early_stopping_mode,
        early_stopping_monitor,
        model_checkpointing,
        model_checkpointing_dir,
        callbacks,
    ):
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
