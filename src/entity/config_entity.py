import os

from pymongo import MongoClient

from dataclasses import dataclass

from src.constants.training_pipeline import *
# from src.constants.database import *

from datetime import datetime



TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")



@dataclass
class TrainingPipelineConfig:
    """
    Configuration class for training pipeline settings.
    """
    pipeline_name: str = PIPELINE_NAME
    artifacts_dir:str = os.path.join(PIPELINE_NAME, ARTIFACTS_DIR, TIMESTAMP)
    timestamp:str = TIMESTAMP

training_pipeline_config = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion settings.
    """
    data_ingestion_dir:str =os.path.join(training_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME) 
    feature_store_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME
    ) 
    
    ingested_data_dir:str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR)
    training_file_path:str = os.path.join(data_ingestion_dir, DATA_INGESTION_DIR_NAME, TRAIN_FILE_NAME)
    testing_file_path:str = os.path.join(data_ingestion_dir, DATA_INGESTION_DIR_NAME, TEST_FILE_NAME)
    
    train_test_split_ratio:float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME
    

    
    
    