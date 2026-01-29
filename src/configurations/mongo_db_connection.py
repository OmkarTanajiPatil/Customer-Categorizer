import os
import sys


import pymongo


from src.constants.database import DATABASE_NAME
from src.constants.env_variables import MONGODB_URL_KEY

from src.exception import CustomException


class MongoDBCLient:
    client = None
    
    def __init__(self, database_name = DATABASE_NAME) -> None:
        try:
            if MongoDBCLient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment key : {MONGODB_URL_KEY} is not set.")
                MongoDBCLient.client = pymongo.MongoClient(mongo_db_url)
                self.client = MongoDBCLient.client
                self.database = self.client[database_name]
                self.database_name = database_name
        except Exception as e:
            raise CustomException(e, sys)
        
    
         