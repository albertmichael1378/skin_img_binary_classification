
# 1. Initializing the environment FIRST
from vgg16classifier.utils.common import set_deterministic_environment
set_deterministic_environment(seed=42)

import tensorflow as tf
from pathlib import Path
from vgg16classifier.entity.configaration_entities import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        inputs = tf.keras.Input(shape=self.config.params_image_size)
        x = inputs
        if self.config.params_is_augmentation:
            x = tf.keras.Sequential(
                [
                    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                    tf.keras.layers.RandomRotation(15 / 360),
                    tf.keras.layers.RandomZoom(0.1),
                    tf.keras.layers.RandomContrast(0.1),
                ],
                name="augmentation",
            )(x)
        preprocessed_inputs = tf.keras.layers.Lambda(
            tf.keras.applications.vgg16.preprocess_input,
            name="vgg16_preprocess",
        )(x)
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
            input_tensor=preprocessed_inputs,
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False
        x = tf.keras.layers.GlobalAveragePooling2D(
            name="head_global_average_pooling"
        )(model.output)
        x = tf.keras.layers.Dense(
            units=256,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="head_dense",
        )(x)
        x = tf.keras.layers.Dropout(rate=0.5, name="head_dropout")(x)
        if classes == 2:
            x = tf.keras.layers.Dense(
                units=1, activation='sigmoid', name="classifier"
            )(x)
        else:
            # If you ever change to multi-class, it automatically switches to softmax
            x = tf.keras.layers.Dense(
                units=classes, activation='softmax', name="classifier"
            )(x)
        # prediction = tf.keras.layers.Dense(
        #     units=classes,
        #     activation="sigmoid"
        # )(flatten_in)
        

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=x
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model:tf.keras.Model):
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
