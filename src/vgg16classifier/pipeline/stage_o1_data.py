from vgg16classifier import logger
from vgg16classifier.components.data_processing import Data_preprocessing
STAGE_NAME = "Data Processing stage vgg16 skin classification"


def main() -> None:
    Data_preprocessing().extract_embedding_matrix()


if __name__ == "__main__":
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    main()
    logger.info(">>>>>> stage %s completed <<<<<<", STAGE_NAME)