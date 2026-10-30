from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    params_is_augmentation: bool
    params_augmentation_brightness_factor: float
    params_head_dropout_rate: float
    params_head_l2_strength: float

@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path
    early_stopping_patience: int

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    SEED: int
    metadata_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data_dir: Path
    params_epochs: int
    params_fine_tune_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_augmentation_brightness_factor: float
    params_image_size: list
    params_learning_rate: float
    params_fine_tune_learning_rate: float
    params_fine_tune_layers: tuple[str, ...]
    params_head_dropout_rate: float
    params_head_l2_strength: float
    params_early_stopping_patience: int
@dataclass(frozen=True)
class EvaluationConfigs:
    root_dir: Path
    path_of_model: Path
    training_metadata_dir: Path
    weights_archive_path: Path
    test_data_dir: Path
    reconstructed_model_path: Path
    metrics_output_path: Path
    predictions_output_path: Path
    class_names: tuple[str, str]
    positive_class: str
    decision_threshold: float
    bootstrap_samples: int
    confidence_level: float
    class_mapping_verified: bool
    dataset_independence_verified: bool
    seed: int
    params_image_size: list
    params_batch_size: int
