import os

from pymongo import MongoClient

from dataclasses import dataclass

from src.constants.training_pipeline import *

# from src.constants.database import *
from src.utils.main_utils import pj

from datetime import datetime


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    """
    Configuration class for training pipeline settings.
    """

    pipeline_name: str = PIPELINE_NAME
    artifacts_dir: str = pj(PIPELINE_NAME, ARTIFACTS_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion settings.
    """

    data_ingestion_dir: str = pj(
        training_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME
    )
    feature_store_file_path: str = pj(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME
    )

    ingested_data_dir: str = pj(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR)
    training_file_path: str = pj(
        data_ingestion_dir, DATA_INGESTION_DIR_NAME, TRAIN_FILE_NAME
    )
    testing_file_path: str = pj(
        data_ingestion_dir, DATA_INGESTION_DIR_NAME, TEST_FILE_NAME
    )

    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME


@dataclass
class DataValidationConfig:
    data_validation_dir: str = pj(
        training_pipeline_config.artifacts_dir, DATA_VALIDATION_DIR_NAME
    )

    valid_data_dir: str = pj(data_validation_dir, DATA_VALIDATION_VALID_DIR)
    invalid_data_dir: str = pj(data_validation_dir, DATA_VALIDATION_INVALID_DIR)

    valid_train_file_path: str = pj(valid_data_dir, TRAIN_FILE_NAME)
    valid_test_file_path: str = pj(valid_data_dir, TEST_FILE_NAME)
    invalid_train_file_path: str = pj(invalid_data_dir, TRAIN_FILE_NAME)
    invalid_test_file_path: str = pj(invalid_data_dir, TEST_FILE_NAME)


@dataclass
class DataTransformationConfig:
    """
    Configuration class for Data Transformation settings.
    """

    data_transformation_dir: str = pj(
        training_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME
    )
    transformed_train_file_path: str = pj(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DIR,
        TRAIN_FILE_NAME.replace(".csv", "npy"),
    )
    transformed_test_file_path: str = pj(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_DIR,
        TEST_FILE_NAME.replace(".csv", "npy"),
    )
    transformed_object_file_path: str = pj(
        data_transformation_dir,
        DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
        PREPROCESSING_OBJECT_FILE_NAME,
    )


class PCAConfig:
    def __init__(self):
        self.n_components = 2
        self.random_state = 42
    
    def get_pca_config(self):
        return self.__dict__
    

class ClusteringConfig:
    def __init__(self):
        self.n_clusters = 3
        self.affinity = 'euclidean'
        self.linkage = 'ward'

    def get_clustering_config(self):
        return self.__dict__


class SimpleImputerConfig:
    def __init__(self):
        self.strategy = 'constant'
        
        self.fill_value = 0
        
    def get_simple_imputer_config(self):
        return self.__dict__


