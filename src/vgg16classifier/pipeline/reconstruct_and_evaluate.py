"""Rebuild the classifier from a Keras weights archive and run evaluation."""

import argparse
from dataclasses import replace
from pathlib import Path

from vgg16classifier import logger
from vgg16classifier.components.evaluation import Evaluation
from vgg16classifier.components.prepare_base_model import PrepareBaseModel
from vgg16classifier.config.configuration import ConfigurationManager


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the project's VGG16 classifier from a Keras weights archive "
            "and evaluate it on the configured labeled dataset."
        )
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("artifacts/training/vgg16model..zip"),
        help="ZIP containing config.json, metadata.json, and model.weights.h5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training/vgg16model_reconstructed.keras"),
        help="Destination for the reconstructed model; existing files are preserved",
    )
    args = parser.parse_args()

    configuration = ConfigurationManager()
    prepare_base_model = PrepareBaseModel(
        config=configuration.get_prepare_base_model_config()
    )
    model_path = prepare_base_model.reconstruct_from_keras_archive(
        archive_path=args.archive,
        output_path=args.output,
    )

    # Evaluate the restored model without changing the normal training model path.
    evaluation_config = replace(
        configuration.get_evaluation_config(),
        path_of_model=model_path,
    )
    evaluation = Evaluation(evaluation_config)
    evaluation.evaluation()
    evaluation.save_score()
    logger.info(
        "Reconstruction and evaluation completed; model saved to %s",
        model_path,
    )


if __name__ == "__main__":
    main()
