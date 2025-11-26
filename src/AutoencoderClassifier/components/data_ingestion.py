import os
import zipfile
from AutoencoderClassifier import logger
import subprocess
from AutoencoderClassifier.exception import CustomException
import sys
from AutoencoderClassifier.utils.common import get_size
from AutoencoderClassifier.entity.config_entity import (DataIngestionConfig)
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd 



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def fetch_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 

            data_source_file = os.path.join('data') #check .dvc file -> path which is your remote path ex.(data)
        
        # --- CRITICAL SAFEGUARD: Pull data if missing ---
            if not os.path.exists(data_source_file):
                logger.info(f"Data file folder'{data_source_file}' missing. Attempting DVC pull...")
                try:
                    logger.run(['dvc', 'pull', data_source_file], check=True)
                    logger.info("DVC pull successful. Data restored.")
                    logger.info("ingestion of data is completed .")
                except subprocess.CalledProcessError as e:
                    logger.error(f"FATAL: DVC pull failed for source data: {e}")
                    raise CustomException(f"DVC pull failed: {e}", sys)

        except Exception as e:
            raise e
    
   
        

from AutoencoderClassifier.config.configuration import ConfigurationManager

if __name__ == "__main__":
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    

    data_ingestion.fetch_file()