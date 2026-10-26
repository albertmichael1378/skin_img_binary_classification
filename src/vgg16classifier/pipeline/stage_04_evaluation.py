from vgg16classifier import logger
from vgg16classifier.config.configuration import ConfigurationManager
from vgg16classifier.components.evaluation import Evaluation
from vgg16classifier.components.prepare_base_model import PrepareBaseModel
STAGE_NAME = "Evaluation stage vgg16 skin classification"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()

        # Rebuild the TensorFlow/Keras 2 model graph and restore tensors from
        # the Keras 3 archive before passing the model to normal evaluation.
        prepare_base_model = PrepareBaseModel(
            config=config.get_prepare_base_model_config()
        )
        prepare_base_model.reconstruct_from_keras_archive(
            archive_path=evaluation_config.weights_archive_path,
            output_path=evaluation_config.reconstructed_model_path,
            overwrite=True,
        )

        evaluation = Evaluation(evaluation_config)
        evaluation.evaluation()
        evaluation.save_score()



if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e