from vgg16classifier import logger
from vgg16classifier.config.configuration import ConfigurationManager
from vgg16classifier.components.data_processing import Data_preprocessing
STAGE_NAME = "Data Processing stage vgg16 skin classification"


if __name__ == "__main__":
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    Data_preprocessing=Data_preprocessing()
    Duplicates=Data_preprocessing.extract_embedding_matrix()
    logger.info(">>>>>> stage %s completed <<<<<<", STAGE_NAME)