import numpy as np
import tensorflow as tf
from keras.applications import ResNet50
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from vgg16classifier import logger
from vgg16classifier.entity.configaration_entities import TrainingConfig


class Data_preprocessing:
    """Extract ResNet50 embeddings and report highly similar image pairs."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(
                "Detected %d GPU device(s); using batch size 64 for embedding extraction",
                len(gpus),
            )
            self.batch_size = 64
        else:
            logger.info(
                "No GPU detected; using batch size 32 for CPU embedding extraction"
            )
            self.batch_size = 32

        image_folder1 = Path(config.training_data_dir)/"cancerous"
        image_folder2 = Path(config.training_data_dir)/"non_cancerous"
        save_folder = Path("artifacts/image_test")
        save_folder.mkdir(parents=True, exist_ok=True)
        self.save_path = save_folder / "resnet_embeddings.npz"

        image_paths1 = list(image_folder1.glob("*.jpg"))
        image_paths2 = list(image_folder2.glob("*.jpg"))
        self.path_strings = [str(p) for p in image_paths1 + image_paths2]
        logger.info(
            "Discovered %d .jpg image(s): %d in %s and %d in %s",
            len(self.path_strings),
            len(image_paths1),
            image_folder1,
            len(image_paths2),
            image_folder2,
        )
        if not self.path_strings:
            raise FileNotFoundError(
                "No .jpg images found in either class folder: "
                f"{image_folder1} or {image_folder2}"
            )

    def load_and_preprocess(self, path):
        """Loads and formats image tensors efficiently."""
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        # Apply ResNet50's specific mathematical scaling
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img

    def extract_embedding_matrix(self):
        """Save image embeddings, then log pairs above the similarity threshold."""
        dataset = tf.data.Dataset.from_tensor_slices(self.path_strings)
        dataset = dataset.map(
            lambda path: self.load_and_preprocess(path),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        logger.info(
            "Loading ResNet50 with ImageNet weights for %d image(s)",
            len(self.path_strings),
        )
        model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

        logger.info("Extracting embeddings with batch size %d", self.batch_size)
        embedding_matrix = model.predict(dataset)
        logger.info("Extracted embeddings with shape %s", embedding_matrix.shape)

        logger.info("Saving embeddings and corresponding image paths to %s", self.save_path)
        np.savez(self.save_path, matrix=embedding_matrix, paths=self.path_strings)

        # 6. Calculate Similarity & Find Duplicates
        print("Calculating similarities...")
        similarity_matrix = cosine_similarity(embedding_matrix)

        threshold = 0.98
        duplicates_found = set()

        logger.info(
            "Comparing image embeddings for duplicate candidates (cosine similarity > %.2f)",
            threshold,
        )
        for i in range(len(self.path_strings)):
            for j in range(i + 1, len(self.path_strings)):
                if similarity_matrix[i, j] > threshold:
                    logger.info(
                        "Duplicate candidate (similarity=%.3f): %s <-> %s",
                        similarity_matrix[i, j],
                        self.path_strings[i],
                        self.path_strings[j],
                    )
                    duplicates_found.add(self.path_strings[j])

        logger.info(
            "Duplicate scan complete: found %d candidate image(s); no files were removed",
            len(duplicates_found),
        )