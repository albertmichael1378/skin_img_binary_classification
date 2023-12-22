from vgg16classifier import logger
from vgg16classifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from vgg16classifier.pipeline.stage_03_training import vgg16_ModelTrainingPipeline
from vgg16classifier.pipeline.stage_04_evaluation import EvaluationPipeline

STAGE_NAME = "Evaluation stage skin disease classification(com726)"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evalution = EvaluationPipeline()
   model_evalution.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
        logger.exception(e)
        raise e
