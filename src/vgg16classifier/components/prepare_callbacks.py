import os
import tensorflow as tf
import time
from vgg16classifier.entity.configaration_entities import PrepareCallbacksConfig
from pathlib import Path

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    def _create_ckpt_callbacks(self, phase: str):
        checkpoint_path = self.config.checkpoint_model_filepath
        phase_checkpoint_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_{phase}{checkpoint_path.suffix}"
        )
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(phase_checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        )

    def get_tb_ckpt_callbacks(self, phase: str):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_{phase}_{timestamp}",
        )
        return [
            tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc",
                mode="max",
                factor=0.3,
                patience=2,
                min_lr=1e-7,
            ),
            self._create_ckpt_callbacks(phase),
        ]
        
   