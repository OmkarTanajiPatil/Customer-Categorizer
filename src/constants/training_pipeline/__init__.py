# Pipeline name and root directory constant

import os


PIPELINE_NAME: str = "src"
ARTIFACTS_DIR: str = "artifacts"
LOG_DIR: str = "logs"
LOG_FILE = "customer_categorization.log"




# COMMON FILES
FILE_NAME: str = "customers.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"
MODEL_FILE_NAME: str = "model.pkl"
SCHEMA_FILE_PATH: str = os.path.join("config", "schema.yaml")



# Data Ingestion constants
DATA_INGESTION_COLLECTION_NAME:str = ""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Config 