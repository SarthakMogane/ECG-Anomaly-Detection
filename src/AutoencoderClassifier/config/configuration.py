from AutoencoderClassifier.utils.common import read_yaml, create_directories,save_json

from AutoencoderClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig)

from AutoencoderClassifier.constants import *
class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            ptbdb_normal_data_path= config.ptbdb_normal_data_path ,
            ptbdb_abnormal_data_path= config.ptbdb_abnormal_data_path
        )

        return data_ingestion_config
    


    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            params_input_dimension=self.params.input_dimension,
            params_latent_dimension=self.params.latent_dimension,
            params_learning_rate=self.params.learning_rate,
           
        )

        return prepare_base_model_config
    
    def get_training_configs(self) -> TrainingConfig:
        training_data = self.config.data_ingestion
        base_model = self.config.prepare_base_model
        training = self.config.training 

        create_directories([training.root_dir])

        prepare_training_configs= TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path= Path(training.trained_model_path),
            base_model_path= Path(base_model.base_model_path),
            training_data = Path(training_data.ptbdb_normal_data_path),
            params_epochs = self.params.epochs,
            params_batch_size = self.params.batch_size,
            params_patience= self.params.patience,
            params_min_delta= self.params.min_delta,
            params_monitor= self.params.monitor,
            params_restore_best_weights = self.params.restore_best_weights
        )

        return prepare_training_configs