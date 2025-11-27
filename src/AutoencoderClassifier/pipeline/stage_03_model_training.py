from AutoencoderClassifier.config.configuration import ConfigurationManager
from AutoencoderClassifier.components.model_training import Training
from AutoencoderClassifier import logger



STAGE_NAME = "Training"



class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_configs()
        training = Training(config=training_config)
        training.get_base_model()
        training.get_data()
        training.split_data()
        training.fit_model()



if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        