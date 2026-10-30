"""Run evaluation by loading the configured native Keras 3 model directly."""

from vgg16classifier import logger
from vgg16classifier.components.evaluation_keras3 import Keras3Evaluation
from vgg16classifier.config.configuration import ConfigurationManager


STAGE_NAME = "Keras 3 skin classifier evaluation"


def main() -> None:
    evaluation_config = ConfigurationManager().get_evaluation_config()
    evaluation = Keras3Evaluation(evaluation_config)
    evaluation.evaluation()
    evaluation.save_score()


if __name__ == "__main__":
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    main()
    logger.info(">>>>>> stage %s completed <<<<<<", STAGE_NAME)
