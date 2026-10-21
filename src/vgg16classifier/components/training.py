
# 1. Initialize environment FIRST
from vgg16classifier.utils.common import set_deterministic_environment
set_deterministic_environment(seed=42)
import tensorflow as tf
from pathlib import Path
import numpy as np
from vgg16classifier import logger
from vgg16classifier.components.run_metadata import (
    create_training_metadata,
    finish_training_metadata,
    write_run_metadata,
)
from vgg16classifier.entity.configaration_entities import TrainingConfig

import os
import numpy as np
from pathlib import Path

# ==========================================
# 1. SET ENVIRONMENT BEFORE TENSORFLOW
# ==========================================
from vgg16classifier.utils.common import set_deterministic_environment
set_deterministic_environment(seed=42)

# ==========================================
# 2. NOW IMPORT TENSORFLOW AND COMPONENTS
# ==========================================
import tensorflow as tf
from vgg16classifier import logger
from vgg16classifier.components.run_metadata import (
    create_training_metadata,
    finish_training_metadata,
    write_run_metadata,
)
from vgg16classifier.entity.configaration_entities import TrainingConfig


def get_balanced_class_weights(labels):
    """
    Calculates balanced class weights identical to sklearn's compute_class_weight.
    """
    labels = np.array(labels).astype(int)
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(np.unique(labels))
    
    class_weights = {}
    for class_id, count in enumerate(class_counts):
        if count > 0:
            class_weights[class_id] = total_samples / (num_classes * count)
    return class_weights


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        # 1. Load Validation Dataset
        self.valid_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.20,
            subset="validation",
            seed=self.config.SEED,
            image_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            label_mode='binary'
        )

        # 2. Load Training Dataset
        self.train_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.20,
            subset="training",
            seed=self.config.SEED,
            image_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            label_mode='binary'
        )
        # BUG FIX: Safely extract perfect, unshuffled labels based on the static file paths
          # 3. Save class mapping
# --------------------------------------

        class_names = self.train_dataset.class_names

        self.class_indices = {
            name: index
            for index, name in enumerate(class_names)
        }      
         # Save file paths BEFORE map() / prefetch()
        logger.info("Saving file paths BEFORE map() / prefetch() as list, inside our object")
        self.train_filepaths = list(self.train_dataset.file_paths)
        self.valid_filepaths = list(self.valid_dataset.file_paths)


        # Extract exact labels from file paths
        logger.info("Extracting exact labels from file paths...")

        self.train_labels = [
            self.class_indices[Path(path).parent.name]
            for path in self.train_filepaths
        ]

        self.valid_labels = [
            self.class_indices[Path(path).parent.name]
            for path in self.valid_filepaths
        ]

        # 3. Apply Data Augmentation
        if self.config.params_is_augmentation:
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(factor=40/360),
                tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
                tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2)
            ])
            
            # Apply augmentation only to training data
            self.train_dataset = self.train_dataset.map(
                lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # 4. Prefetching for GPU optimization
        self.train_dataset = self.train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.valid_dataset = self.valid_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    def save_weights(self, path: Path, model: tf.keras.Model):
        model.save_weights(path)

    def train(self, callback_list: list):
        repo_root = Path(__file__).resolve().parents[3]
        # In configuration.py
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
            self.train_dataset = Path(self.config.training.kaggle_data_dir)
        else:
            self.train_dataset = Path(self.config.training.training_data_dir)
            
        
        # Calculate balanced weights
        class_weight_dict = get_balanced_class_weights(self.train_labels)
        logger.info(f"Class weights: {class_weight_dict}")

        # BUG FIX: Using exactly your configured metadata path
        metadata = create_training_metadata(
            repo_root=repo_root,
            dataset_root=Path(self.config.training_data),
            train_filenames=self.train_filepaths,
            train_labels=self.train_labels,
            validation_filenames=self.valid_filepaths,
            validation_labels=self.valid_labels,
            class_indices=self.class_indices,
            parameters={
                "class_weight": class_weight_dict,
                "SEED": self.config.SEED,
                "epochs": self.config.params_epochs,
                "batch_size": self.config.params_batch_size,
                "augmentation": self.config.params_is_augmentation,
                "image_size": list(self.config.params_image_size),
                "validation_split": 0.20,
                "learning_rate": float(
                    tf.keras.backend.get_value(self.model.optimizer.learning_rate)
                ),
            },
            preprocessing={
                "target_size": list(self.config.params_image_size[:-1]),
                "color_mode": "rgb",
                "rescale": None, 
                "vgg16_preprocess_input": False,
            },
            optimizer=(
                type(self.model.optimizer).__name__
                if self.model.optimizer is not None
                else None
            ),
            loss=str(self.model.loss),
            model_path=Path(self.config.trained_model_path),
        )
        write_run_metadata(self.config.metadata_path, metadata)

        try:
            history = self.model.fit(
                self.train_dataset,
                epochs=self.config.params_epochs,
                validation_data=self.valid_dataset,
                callbacks=callback_list,
                class_weight=class_weight_dict,
            )
            # Change the extension to match Keras 3 weight standards
            weights_path = self.config.trained_model_path.replace(".keras", ".weights.h5")
            self.model.save_weights(weights_path)
            #self.save_model(
                #path=self.config.trained_model_path,
                #model=self.model,
            #)
        except Exception as error:
            finish_training_metadata(
                metadata,
                status="failed",
                error=error,
            )
            try:
                write_run_metadata(self.config.metadata_path, metadata)
            except OSError:
                logger.exception("Could not persist failed training-run metadata")
            raise

        finish_training_metadata(
            metadata,
            status="completed",
            history=history.history,
            model_path=Path(self.config.trained_model_path),
        )
        write_run_metadata(self.config.metadata_path, metadata)

########New optimized code for speed on GPU using tf.data.Dataset and avoiding loops for class weight calculation.

class Training_Old:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            class_mode='binary',
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            class_mode='binary',
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    import numpy as np

    def get_balanced_class_weights(labels):
        """
        Calculates balanced class weights identical to sklearn's compute_class_weight.
        """
        # Count how many times each class appears
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        num_classes = len(np.unique(labels))
        
        # Calculate the weight for each class
        class_weights = {}
        for class_id, count in enumerate(class_counts):
            if count > 0: # Avoid division by zero
                class_weights[class_id] = total_samples / (num_classes * count)
                
        return class_weights

    def train(self, callback_list: list):
        if self.train_generator.class_indices != self.valid_generator.class_indices:
            raise ValueError("Training and validation class mappings do not match")

        repo_root = Path(__file__).resolve().parents[3]
        metadata_path = Path(self.config.trained_model_path).with_name(
            "run_metadata.json"
        )
        class_indices = dict(self.train_generator.class_indices)
        metadata = create_training_metadata(
            repo_root=repo_root,
            dataset_root=Path(self.config.training_data),
            train_filenames=self.train_generator.filenames,
            train_labels=self.train_generator.classes,
            validation_filenames=self.valid_generator.filenames,
            validation_labels=self.valid_generator.classes,
            class_indices=class_indices,
            parameters={
                "class_weight": "class_weight_dict",
                "SEED": self.config.params_seed,
                "epochs": self.config.params_epochs,    
                "batch_size": self.config.params_batch_size,
                "augmentation": self.config.params_is_augmentation,
                "image_size": list(self.config.params_image_size),
                "validation_split": 0.20,
                "learning_rate": float(
                    tf.keras.backend.get_value(self.model.optimizer.learning_rate)
                ),
            },
            preprocessing={
                "target_size": list(self.config.params_image_size[:-1]),
                "color_mode": "rgb",
                "rescale": None,
                "vgg16_preprocess_input": False,
            },
            optimizer=(
                type(self.model.optimizer).__name__
                if self.model.optimizer is not None
                else None
            ),
            loss=str(self.model.loss),
            model_path=Path(self.config.trained_model_path),
        )
        write_run_metadata(self.config.metadata_path, metadata)
        # 1. Extract labels from the tf.data.Dataset
        # (Assuming your dataset is named 'train_dataset')
        train_labels = np.concatenate([y.numpy() for x, y in self.train_generator], axis=0)

        # If your labels are one-hot encoded (e.g., shape is (batch_size, 2)), convert to 1D:
        if len(train_labels.shape) > 1 and train_labels.shape[1] > 1:
            train_labels = np.argmax(train_labels, axis=1)
        # 2. Compute the balanced class weights
        # 3. Convert to dictionary
        # INSTANT: No looping required
        labels = self.train_generator.classes
        try:
            history = self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                validation_data=self.valid_generator,
                callbacks=callback_list,
                class_weight=get_balanced_class_weights(labels),
            )
            self.save_model(
                path=self.config.trained_model_path,
                model=self.model,
            )
        except Exception as error:
            finish_training_metadata(
                metadata,
                status="failed",
                error=error,
            )
            try:
                write_run_metadata(self.config.metadata_path, metadata)
            except OSError:
                logger.exception("Could not persist failed training-run metadata")
            raise

        finish_training_metadata(
            metadata,
            status="completed",
            history=history.history,
            model_path=Path(self.config.trained_model_path),
        )
        write_run_metadata(self.config.metadata_path, metadata)