from vgg16classifier import logger
from vgg16classifier.components.data_processing import Data_preprocessing
from vgg16classifier.config.configuration import ConfigurationManager

STAGE_NAME = "Data Processing stage vgg16 skin classification"


def main() -> None:
    """Run the duplicate-image embedding and candidate-reporting stage."""
    config = ConfigurationManager()
    training_config = config.get_training_config()
    Data_preprocessing(training_config).extract_embedding_matrix()


if __name__ == "__main__":
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    try:
        main()
    except Exception:
        logger.exception("Stage %s failed", STAGE_NAME)
        raise
    else:
        logger.info(">>>>>> stage %s completed <<<<<<", STAGE_NAME)