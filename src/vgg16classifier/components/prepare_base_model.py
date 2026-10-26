
# 1. Initializing the environment FIRST
from vgg16classifier.utils.common import set_deterministic_environment
set_deterministic_environment(seed=42)

import os
import shutil
import tempfile
import zipfile

import tensorflow as tf
from pathlib import Path
from vgg16classifier.entity.configaration_entities import PrepareBaseModelConfig


@tf.keras.utils.register_keras_serializable(package="vgg16classifier")
class VGG16Preprocessing(tf.keras.layers.Layer):
    """Apply the standard VGG16 RGB-to-BGR preprocessing before the backbone."""

    def call(self, inputs):
        return tf.keras.applications.vgg16.preprocess_input(inputs)


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def _build_base_model(self, weights: str | None) -> None:
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
        preprocessed_inputs = VGG16Preprocessing(name="vgg16_preprocess")(x)
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=weights,
            include_top=self.config.params_include_top,
            input_tensor=preprocessed_inputs,
        )

    def get_base_model(self):
        self._build_base_model(weights=self.config.params_weights)
        self.save_model(path=self.config.base_model_path, model=self.model)

    def reconstruct_from_keras_archive(
        self,
        archive_path: Path,
        output_path: Path,
        overwrite: bool = False,
    ) -> Path:
        """Rebuild this project's classifier and import tensors from a Keras 3 archive."""
        archive_path = Path(archive_path)
        output_path = Path(output_path)
        if not archive_path.is_file():
            raise FileNotFoundError(f"Model weights archive does not exist: {archive_path}")
        if output_path.suffix.lower() != ".keras":
            raise ValueError(f"Reconstructed model must use .keras format: {output_path}")
        if archive_path.resolve() == output_path.resolve():
            raise ValueError("The reconstructed model cannot overwrite its source archive")
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite an existing model: {output_path}"
            )

        # The original archive's Keras 3 Functional config cannot load in this
        # project's Keras 2 runtime, so rebuild the known VGG16 classifier graph.
        self._build_base_model(weights=None)
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
        )
        self._load_keras3_weights(self.full_model, archive_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="vgg16-reconstruction-",
            dir=output_path.parent,
        ) as temp_dir:
            temporary_model_path = Path(temp_dir) / output_path.name
            self.save_model(temporary_model_path, self.full_model)
            os.replace(temporary_model_path, output_path)
        return output_path

    @staticmethod
    def _load_keras3_weights(model: tf.keras.Model, archive_path: Path) -> None:
        import h5py

        required_members = {"config.json", "metadata.json", "model.weights.h5"}
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.namelist()
            if len(members) != len(set(members)):
                raise ValueError("Model archive contains duplicate file entries")
            missing_members = required_members.difference(members)
            if missing_members:
                raise ValueError(
                    "Model archive is missing required root files: "
                    f"{sorted(missing_members)}"
                )
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise ValueError(f"Corrupt file in model archive: {corrupt_member}")

            # Extract only the weight file to a temporary location; never unpack
            # archive-provided paths into the project workspace.
            with tempfile.TemporaryDirectory(prefix="vgg16-weight-restore-") as temp_dir:
                weight_path = Path(temp_dir) / "model.weights.h5"
                with archive.open("model.weights.h5") as source:
                    with weight_path.open("wb") as destination:
                        shutil.copyfileobj(source, destination)

                with h5py.File(weight_path, "r") as weights_file:
                    if "layers" not in weights_file:
                        raise ValueError("Keras weight file has no layer weights")

                    saved_layers: dict[str, h5py.Group] = {}
                    for layer_group in weights_file["layers"].values():
                        variables = layer_group.get("vars")
                        if variables is None or not len(variables):
                            continue
                        saved_name = variables.attrs.get("name")
                        if isinstance(saved_name, bytes):
                            saved_name = saved_name.decode("utf-8")
                        if not isinstance(saved_name, str) or not saved_name:
                            raise ValueError(
                                "Keras weight file contains a layer without a name"
                            )
                        if saved_name in saved_layers:
                            raise ValueError(
                                f"Duplicate saved weights for layer {saved_name!r}"
                            )
                        saved_layers[saved_name] = variables

                    model_layers = {
                        layer.name: layer for layer in model.layers if layer.weights
                    }
                    if set(saved_layers) != set(model_layers):
                        raise ValueError(
                            "The archive and reconstructed model have different "
                            "weighted layers. "
                            f"Missing: {sorted(set(model_layers) - set(saved_layers))}; "
                            f"unexpected: {sorted(set(saved_layers) - set(model_layers))}"
                        )

                    assignments: list[tuple[tf.keras.layers.Layer, list]] = []
                    for layer_name, layer in model_layers.items():
                        variables = saved_layers[layer_name]
                        variable_indices = sorted(
                            (int(key) for key in variables.keys())
                        )
                        if variable_indices != list(range(len(layer.weights))):
                            raise ValueError(
                                f"Unexpected weight variables for layer {layer_name!r}"
                            )
                        values = [
                            variables[str(index)][()]
                            for index in variable_indices
                        ]
                        expected_shapes = [
                            tuple(weight.shape) for weight in layer.get_weights()
                        ]
                        actual_shapes = [tuple(value.shape) for value in values]
                        if actual_shapes != expected_shapes:
                            raise ValueError(
                                f"Weight shape mismatch for {layer_name!r}: "
                                f"archive {actual_shapes}, model {expected_shapes}"
                            )
                        assignments.append((layer, values))

                    # Validate every layer before mutating the model.
                    for layer, values in assignments:
                        layer.set_weights(values)

    def _prepare_full_model(self, model, classes, freeze_all, freeze_till):
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
            kernel_regularizer=tf.keras.regularizers.l2(
                self.config.params_head_l2_strength
            ),
            name="head_dense",
        )(x)
        x = tf.keras.layers.Dropout(
            rate=self.config.params_head_dropout_rate,
            name="head_dropout",
        )(x)
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
        Path(path).parent.mkdir(parents=True, exist_ok=True)
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
