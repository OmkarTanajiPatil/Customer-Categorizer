import sys

from typing import Optional

import numpy as np
import pandas as pd

from src.configurations.mongo_db_connection import MongoDBCLient

from src.constants.database import DATABASE_NAME,  COLLECTION_NAME

from src.exception import CustomException

class CustomerData:
    '''
    This class helps to export entire mongo db record as a pandas dataframe
    
    '''
    
    def __init__(self):
        """"""
        
        try:
            self.mongo_client = MongoDBCLient(database_name=DATABASE_NAME)
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def export_collection_as_dataframe(
        self, collection_name:str, database_name: Optional[str] = None
    ) -> pd.DataFrame:
        try:
            if database_name is None:
                collection = self.mongo_client.database[COLLECTION_NAME]
            else:
                collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.tolist():
                df = df.drop(columns=['_id'] , axis=1)
            df.replace({"na": np.nan}, inplace=True)
            return df
            
        
        except Exception as e:
            raise CustomException(e, sys)
    