from AutoencoderClassifier.config.configuration import ConfigurationManager
from AutoencoderClassifier.components.model_evaluation import Evaluation
from AutoencoderClassifier import logger
from dotenv import load_dotenv
load_dotenv()



STAGE_NAME = "Evaluation"



class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluator = Evaluation(config=evaluation_config)
        evaluator.run_evaluation()



if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    