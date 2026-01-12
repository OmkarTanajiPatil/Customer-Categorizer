from src.components.mongo_connection import create_mongo_connection
import os
from src.utils.exception import CustomException
from src.utils.logger import logging
import sys

import pandas as pd


def ingest_data_from_mongo():
    """
    Importing data from MongoDB collection and returning it as a list of dictionaries.
    """
    try:
        mongo_collection_name = os.getenv("MONGO_COLLECTION_NAME")
        
        # Connection
        client = create_mongo_connection()
        db = client.get_database()

        # Data
        customers_collection = db[mongo_collection_name]
        data = list(customers_collection.find())
        logging.info(f"Data imported from MongoDB collection: {mongo_collection_name} successfully.")
        return data
    except Exception as e:
        logging.error(f"Error occurred while importing data from MongoDB: {e}")
        raise CustomException(e,sys)


def store_data_to_parquet():
    """
    Ingest data from MongoDB and store it as a Parquet file.
    """
    try:
        file_path = os.getenv("MONGO_INGESTED_DATA_LOCATION")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        data = ingest_data_from_mongo()
        df = pd.DataFrame(data)
        df["_id"] = df["_id"].astype(str)
        df.to_parquet(file_path, index=False)
        logging.info(f"Data stored to {file_path} successfully.")
    except Exception as e:
        logging.error(f"Error occurred while storing data to Parquet: {e}")
        raise CustomException(e,sys)
    
if __name__ == "__main__":
    store_data_to_parquet()
    