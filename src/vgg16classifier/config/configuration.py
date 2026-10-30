from pathlib import Path

from vgg16classifier.constants import*
from vgg16classifier.utils.common import read_yaml, create_directories
from vgg16classifier.entity.configaration_entities import (PrepareBaseModelConfig,PrepareCallbacksConfig,TrainingConfig,EvaluationConfigs)
import os

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
            config = self.config.prepare_base_model

            create_directories([config.root_dir])

            prepare_base_model_config = PrepareBaseModelConfig(
                root_dir=Path(config.root_dir),
                base_model_path=Path(config.base_model_path),
                updated_base_model_path=Path(config.updated_base_model_path),
                params_image_size=self.params.IMAGE_SIZE,
                params_learning_rate=self.params.LEARNING_RATE,
                params_include_top=self.params.INCLUDE_TOP,
                params_weights=self.params.WEIGHTS,
                params_classes=self.params.CLASSES,
                params_is_augmentation=self.params.AUGMENTATION,
                params_head_dropout_rate=self.params.HEAD_DROPOUT_RATE,
                params_head_l2_strength=self.params.HEAD_L2_STRENGTH,
            )

            return prepare_base_model_config
    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        prepare_callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath),
            early_stopping_patience=self.params.EARLY_STOPPING_PATIENCE,
        )

        return prepare_callback_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        trained_model_path = Path(training.trained_model_path)
        if trained_model_path.suffix.lower() != ".keras":
            raise ValueError(
                "training.trained_model_path must use the native .keras format"
            )
        create_directories([
            Path(training.root_dir),
            Path(training.metadata_dir),
        ])

        training_config = TrainingConfig(
            SEED=self.params.SEED,
            root_dir=Path(training.root_dir),
            metadata_dir=Path(training.metadata_dir),
            trained_model_path=trained_model_path,
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data_dir=Path(training.training_data_dir),
            params_epochs=params.EPOCHS,
            params_fine_tune_epochs=params.FINE_TUNE_EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_fine_tune_learning_rate=params.FINE_TUNE_LEARNING_RATE,
            params_fine_tune_layers=tuple(params.FINE_TUNE_LAYERS),
            params_head_dropout_rate=params.HEAD_DROPOUT_RATE,
            params_head_l2_strength=params.HEAD_L2_STRENGTH,
            params_early_stopping_patience=params.EARLY_STOPPING_PATIENCE,
        )

        return training_config
    def get_evaluation_config(self) -> EvaluationConfigs:
        evaluation = self.config.evaluation
        metrics_output_path = Path(evaluation.metrics_output_path)
        predictions_output_path = Path(evaluation.predictions_output_path)
        # Ensure configured artifact paths exist before evaluation starts.
        create_directories(
            [
                Path(evaluation.root_dir),
                Path(evaluation.reconstructed_model_path).parent,
                metrics_output_path.parent,
                predictions_output_path.parent,
            ]
        )

        return EvaluationConfigs(
            root_dir=Path(evaluation.root_dir),
            # Evaluate the model artifact directly; reconstruction is a separate
            # compatibility workflow for runtimes that cannot load its format.
            path_of_model=Path(self.config.training.trained_model_path),
            training_metadata_dir=Path(self.config.training.metadata_dir),
            weights_archive_path=Path(evaluation.weights_archive_path),
            test_data_dir=Path(evaluation.test_data_dir),
            reconstructed_model_path=Path(evaluation.reconstructed_model_path),
            metrics_output_path=metrics_output_path,
            predictions_output_path=predictions_output_path,
            class_names=tuple(evaluation.class_names),
            positive_class=evaluation.positive_class,
            decision_threshold=evaluation.decision_threshold,
            bootstrap_samples=evaluation.bootstrap_samples,
            confidence_level=evaluation.confidence_level,
            class_mapping_verified=evaluation.class_mapping_verified,
            dataset_independence_verified=evaluation.dataset_independence_verified,
            seed=self.params.SEED,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
        )