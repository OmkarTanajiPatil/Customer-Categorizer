from pymongo import MongoClient
import os
from src.utils.exception import CustomException
from src.utils.logger import logging
import sys

import dotenv

loaded_dot_env = dotenv.load_dotenv()

def create_mongo_connection():
    """
    Creates and returns a MongoDB client connection using environment variables for configuration.
    """
    # Load environment variables
    mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING")

    # Establish MongoDB connection
    uri = f"{mongo_connection_string}"
    try:
        client = MongoClient(uri)
        logging.info("Connected to MongoDB successfully.")
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {e}")
        raise CustomException(e, sys)

    return client


if __name__ == '__main__':
    print(create_mongo_connection())