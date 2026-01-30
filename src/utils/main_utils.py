from src.exception import CustomException
from src.logger import logging

import pandas as pd
import pickle
import numpy as np

import sys

from src.constants.training_pipeline import *

import os

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


def pj(self, *args):
    return os.path.join(*args)


class MainUtils:
    def __init__(self):
        pass

    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomException(e, sys)

    def read_schema_config_file(
        self,
    ) -> dict:
        try:
            schema_config = self.read_yaml_file(SCHEMA_FILE_PATH)

            return schema_config

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def save_object(obj: object, path: str):
        logging.info("Entered the save_object method of class MainUtils")
        try:
            with open(path, "wb") as file_obj:
                pickle.dump(obj, file_obj)

            logging.info("Exited the save_object method of class MainUtils")
        except Exception as e:
            raise CustomException(e, sys)

    def save_numpy_array_data(file_path: str, array: np.array):
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)
        except Exception as e:
            raise CustomException(e, sys)
