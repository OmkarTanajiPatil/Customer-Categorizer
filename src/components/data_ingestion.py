import os
import sys
from typing import Tuple
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.constants.database import DATABASE_NAME, COLLECTION_NAME

from src.data_access.customer_data import CustomerData

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

from src.utils.main_utils import MainUtils


from src.logger import logging
from src.exception import CustomException

class DataIngestion:
    
    def __init__(self, data_ingestion_config:DataIngestionConfig = DataIngestionConfig()):
        self.data_ingestion_config = data_ingestion_config
        self.utils = MainUtils()

    def split_data_as_train_test(self, dataframe:DataFrame) -> Tuple[DataFrame, DataFrame]:
        logging.info("Entered split_data_as_train_test method of DataIngestion class")
        
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            
            logging.info("Performed train test split on the dataframe")
            
            logging.info(
                "Exited split_data_as_train_test method of DataIngestion class"
            )
            
            ingested_data_dir:str = self.data_ingestion_config.ingested_data_dir
            
            os.makedirs(ingested_data_dir, exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index = False, header = True)
            logging.info("Training data is stored")
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index = False, header = True)
            logging.info("Testing data is stored")
            
        
        except Exception as e:
            raise CustomException(e, sys)
    
    
    def export_data_into_feature_store(self) -> DataFrame:
        """
        Reads data from mongo db and store into feature store
        """
        
        try:
            logging.info("Exporting data form mongo")
            customer_data = CustomerData()
            customer_dataframe = customer_data.export_collection_as_dataframe(collection_name=COLLECTION_NAME, database_name=DATABASE_NAME)
            
            logging.info(f"Shape of the dataframe : {customer_dataframe.shape}")
            
            feature_store_file_path:str = self.data_ingestion_config.feature_store_file_path
            dir_path:str = os.path.dirname(feature_store_file_path)
            
            
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            customer_dataframe.to_csv(feature_store_file_path, index= False, header=True)
            return customer_dataframe
        
        except Exception as e:
            raise CustomException(e, sys)
    
    
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        
        logging.info("Entered initiate_data_ingestion method of  DataIngestion class")
        
        try:
            dataframe = self.export_data_into_feature_store()
            
            _schem_config = self.utils.read_schema_config_file()
            
            dataframe.drop(_schem_config["drop_columns"], axis=1)
            
            logging.info("Got the data from mongoDb")
            
            logging.info("Exited initiate_data_ingestion method of DataIngestion class")
            
            dataIngestionArtifact = DataIngestionArtifact(
                self.data_ingestion_config.training_file_path,
                self.data_ingestion_config.testing_file_path
            )
            
            logging.info(f"Data Ingestion artifact: {dataIngestionArtifact}")
            return dataIngestionArtifact
            
        except Exception as e:
            raise CustomException(e, sys)