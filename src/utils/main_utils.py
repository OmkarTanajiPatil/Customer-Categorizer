from src.exception import CustomException
from src.logger import logging

import pandas as pd
import sys

from src.constants.training_pipeline import *


import yaml

def save_df(df: pd.DataFrame, path: str) -> None:
    """
    Save a pandas DataFrame to a specified path in Parquet format.
    """
    try:
        df.to_parquet(path, index=False)
        logging.info(f"DataFrame saved to {path} successfully.")
    except Exception as e:
        logging.error(f"Error occurred while saving DataFrame to {path}: {e}")
        raise CustomException(e, sys)
    

class MainUtils:
    
    def __init__(self):
        pass
    
    def read_yaml_file(self, filename:str) ->dict:
        try:
            with open(filename,  'rb') as yaml_file:
                return yaml.safe_load(yaml_file)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def read_schema_config_file(self, ) -> dict:
        try:
            schema_config = self.read_yaml_file(SCHEMA_FILE_PATH)
            
            return schema_config
            
        except Exception as e:
            raise CustomException(e, sys)
        