# import os
# import json
import sys

import pandas as pd
from pandas import DataFrame
# from evidently import Report
# from evidently.metrics import


from src.logger import logging
from src.exception import CustomException

from src.utils.main_utils import MainUtils

from src.constants.training_pipeline import *  # noqa: F403

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig

from typing import Tuple


class DataValidation:
    def __init__(
        self,
        dataIngestionArtifact: DataIngestionArtifact,
        dataValidationConfig: DataValidationConfig,
    ):
        self.dataIngestionArtifact = dataIngestionArtifact
        self.dataValidationConfig = dataValidationConfig

        self.utils = MainUtils()

        self._schema_config = self.utils.read_schema_config_file()

    def validate_schema_columns(self, dataframe: DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required columns present {status}")

            return status

        except Exception as e:
            raise CustomException(e, sys)

    def validate_dataset_schema_columns(self, train_set, test_set) -> Tuple[bool, bool]:
        logging.info(
            "Entered validate_dataset_schema_columns method of Data_Validation class"
        )

        try:
            logging.info("Validating dataset schema column")

            train_schema_status = self.validate_schema_columns(train_set)

            logging.info("Validated dataset schema columns in train set")

            test_schema_status = self.validate_schema_columns(test_set)

            logging.info("Validated dataset schema columns in test set")

            logging.info(
                "Exited validate_dataset_schema_columns method of Data_Validation class"
            )

            return train_schema_status, test_schema_status
        except Exception as e:
            raise CustomException(e, sys)

    def detect_dataset_drift(
        self, reference_data: DataFrame, currentL_df: DataFrame
    ) -> bool:
        try:
            return False
            pass

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        logging.info("Entered data validation for the dataset")
        try:
            train_df = DataValidation.read_data(
                self.dataIngestionArtifact.trained_file_path
            )
            test_df = DataValidation.read_data(
                self.dataIngestionArtifact.test_file_path
            )

            drift = self.detect_dataset_drift(train_df, test_df)

            (schema_train_col_status, schema_test_col_status) = (
                self.validate_dataset_schema_columns(train_df, test_df)
            )

            logging.info(f"Schema train col status is : {schema_train_col_status}")
            logging.info(f"Schema test col status is : {schema_test_col_status}")

            if schema_train_col_status and schema_test_col_status and not drift:
                logging.info("Dataset schema validation complete")
                validation_status = True
            else:
                validation_status = False

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.dataIngestionArtifact.trained_file_path,
                valid_test_file_path=self.dataIngestionArtifact.test_file_path,
                invalid_test_file_path=self.dataValidationConfig.invalid_train_file_path,
                invalid_train_file_path=self.dataValidationConfig.invalid_test_file_path,
            )

            logging.info("Exited data validation for the dataset")

            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)
