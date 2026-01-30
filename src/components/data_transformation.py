import sys
import os

import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


from src.logger import logging
from src.exception import CustomException


from src.constants.training_pipeline import *  # noqa: F403

from src.utils.main_utils import MainUtils
from src.entity.config_entity import DataTransformationConfig, SimpleImputerConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)

from src.components.data_ingestion import DataIngestion
from src.components.data_clustering import CreateCluster

from datetime import datetime


class DataTransformation:
    def __init__(
        self,
        dataIngestionArtifact: DataIngestionArtifact,
        dataValidationArtifact: DataValidationArtifact,
        dataTransformationConfig: DataTransformationConfig,
    ):
        self.dataIngestionArtifact = dataIngestionArtifact
        self.dataValidationArtifact = dataValidationArtifact
        self.dataTransformationConfig = dataTransformationConfig
        self.dataIngestion = DataIngestion()

        self.imputer_config = SimpleImputerConfig()

        self.utils = MainUtils()

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def get_new_features(self, train_set: DataFrame, test_set: DataFrame) -> DataFrame:
        try:
            logging.info("New feature creation started")
            train_set_with_new_features: DataFrame = DataFrame()
            test_set_with_new_features: DataFrame = DataFrame()

            datasets = {"train_set": train_set, "test_set": test_set}

            for key in datasets:
                dataset = datasets[key]
                # create new column for feature
                ##  creating a new field to store the Age of the customer
                current_year = datetime.now().year

                dataset["Age"] = current_year - dataset["Year_Birth"]

                ###  recoding the customer's education level to numeric form (0: high-school, 1: diploma, 2: bachelors, 3: masters, and 4: doctorates)
                dataset["Education"] = (
                    dataset["Education"]
                    .map(
                        {
                            "Basic": 0,
                            "2n Cycle": 1,
                            "Graduation": 2,
                            "Master": 3,
                            "PhD": 4,
                        }
                    )
                    .astype("int64")
                )

                ###  recoding the customer's marital status to binary form (1: married or together, 0: others)
                dataset["Marital_Status"] = (
                    dataset["Marital_Status"]
                    .map(
                        {
                            "Married": 1,
                            "Together": 1,
                            "Absurd": 0,
                            "Widow": 0,
                            "YOLO": 0,
                            "Divorced": 0,
                            "Single": 0,
                            "Alone": 0,
                        }
                    )
                    .astype("int64")
                )

                #  creating a new field to store the number of children in the household
                dataset["Children"] = dataset["Kidhome"] + dataset["Teenhome"]

                # creating Family_Size
                dataset["Family_Size"] = (
                    dataset["Marital_Status"] + dataset["Children"] + 1
                )

                #  creating a new field to store the total spending of the customer
                dataset["Total_Spending"] = (
                    dataset["MntWines"]
                    + dataset["MntFruits"]
                    + dataset["MntMeatProducts"]
                    + dataset["MntFishProducts"]
                    + dataset["MntSweetProducts"]
                    + dataset["MntGoldProds"]
                )
                dataset["Total Promo"] = (
                    dataset["AcceptedCmp1"]
                    + dataset["AcceptedCmp2"]
                    + dataset["AcceptedCmp3"]
                    + dataset["AcceptedCmp4"]
                    + dataset["AcceptedCmp5"]
                )

                ## The following code works out how long the customer has been with the company and store the total number of promotions the customers responded to
                dataset["Dt_Customer"] = pd.to_datetime(
                    dataset["Dt_Customer"], format="%d-%m-%Y"
                )
                today = datetime.today()
                dataset["Days_as_Customer"] = (today - dataset["Dt_Customer"]).dt.days
                dataset["Offers_Responded_To"] = (
                    dataset["AcceptedCmp1"]
                    + dataset["AcceptedCmp2"]
                    + dataset["AcceptedCmp3"]
                    + dataset["AcceptedCmp4"]
                    + dataset["AcceptedCmp5"]
                    + dataset["Response"]
                )
                dataset["Parental Status"] = np.where(dataset["Children"] > 0, 1, 0)

                # dropping columns which are already used to create new features
                columns_to_drop = ["Year_Birth", "Kidhome", "Teenhome"]
                dataset.drop(columns=columns_to_drop, axis=1, inplace=True)
                dataset.rename(
                    columns={
                        "Marital_Status": "Marital Status",
                        "MntWines": "Wines",
                        "MntFruits": "Fruits",
                        "MntMeatProducts": "Meat",
                        "MntFishProducts": "Fish",
                        "MntSweetProducts": "Sweets",
                        "MntGoldProds": "Gold",
                        "NumWebPurchases": "Web",
                        "NumCatalogPurchases": "Catalog",
                        "NumStorePurchases": "Store",
                        "NumDealsPurchases": "Discount Purchases",
                    },
                    inplace=True,
                )

                dataset = dataset[
                    [
                        "Age",
                        "Education",
                        "Marital Status",
                        "Parental Status",
                        "Children",
                        "Income",
                        "Total_Spending",
                        "Days_as_Customer",
                        "Recency",
                        "Wines",
                        "Fruits",
                        "Meat",
                        "Fish",
                        "Sweets",
                        "Gold",
                        "Web",
                        "Catalog",
                        "Store",
                        "Discount Purchases",
                        "Total Promo",
                        "NumWebVisitsMonth",
                    ]
                ]

                if key == "train_set":
                    train_set_with_new_features = pd.concat(
                        [train_set_with_new_features, dataset]
                    )
                else:
                    test_set_with_new_features = pd.concat(
                        [test_set_with_new_features, dataset]
                    )

            logging.info("New feature creation completed successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def transform_data(self, train_set: DataFrame, test_set: DataFrame) -> DataFrame:
        logging.info("Entered transform_data method of DataTransformation class")
        try:
            logging.info("Got numerical cols from schema config")

            datasets = {"train_set": train_set, "test_set": test_set}

            for key, df in datasets.items():
                df.dropna(inplace=True)
                df.drop_duplicates(inplace=True)

                drop_columns = ["Z_CostContact", "Z_Revenue", "ID", "_id"]
                for dc in drop_columns:
                    if dc in df.columns:
                        df.drop(dc, axis=1, inplace=True)

                datasets[key] = df

            train_set = datasets["train_set"]
            test_set = datasets["test_set"]

            numeric_features: list = [
                feature
                for feature in train_set.columns
                if train_set[feature].dtype != "O"
            ]

            outlier_features = [
                "Wines",
                "Fruits",
                "Meat",
                "Fish",
                "Sweets",
                "Gold",
                "Age",
                "Total_Spending",
            ]

            numeric_features = [
                features
                for features in numeric_features
                if features not in outlier_features
            ]

            logging.info("Initilise StandardScaler and SimpleImputer")

            numeric_pipeline = Pipeline(
                steps=[
                    (
                        "Imputer",
                        SimpleImputer(
                            **self.imputer_config.get_simple_imputer_config()
                        ),
                    ),
                    ("StandardScaler", StandardScaler()),
                ]
            )

            outlier_feature_pipeline = Pipeline(
                steps=[
                    (
                        "Imputer",
                        SimpleImputer(
                            **self.imputer_config.get_simple_imputer_config()
                        ),
                    ),
                    ("Transformer", PowerTransformer(standardize=True)),
                ]
            )

            preprocesser = ColumnTransformer(
                [
                    ("numeric_pipeline", numeric_pipeline, numeric_features),
                    ("outlier_pipeline", outlier_feature_pipeline, outlier_features),
                ]
            )

            preprocessed_train_set = preprocesser.fit_transform(train_set)
            preprocessed_test_set = preprocesser.transform(test_set)

            columns = train_set.columns
            preprocessed_train_set = pd.DataFrame(
                preprocessed_train_set, columns=columns
            )
            preprocessed_test_set = pd.DataFrame(preprocessed_test_set, columns=columns)

            preprocessor_obj_dir = os.path.dirname(
                self.dataTransformationConfig.transformed_object_file_path
            )
            os.makedirs(preprocessor_obj_dir, exist_ok=True)

            self.utils.save_object(
                preprocesser, self.dataTransformationConfig.transformed_object_file_path
            )

            logging.info(
                "Saved preprocessor object at: ",
                self.dataTransformationConfig.transformed_object_file_path,
            )

            logging.info("Exited transform_data method of DataTransformation class")

            return preprocessed_train_set, preprocessed_test_set

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        logging.info(
            "Entered initiate_data_transformation method of DataTransformation class"
        )
        try:
            if self.dataValidationArtifact.validation_status:
                train_set = DataTransformation.read_data(
                    self.dataValidationArtifact.valid_train_file_path
                )
                test_set = DataTransformation.read_data(
                    self.dataValidationArtifact.valid_test_file_path
                )

                train_set, test_set = self.get_new_features(train_set, test_set)

                logging.info("Got the preprocessor object")

                preprocessed_train_set, preprocessed_test_set = self.transform_data(
                    train_set, test_set
                )

                cluster_creator = CreateCluster()

                labelled_train_set = cluster_creator.initilize_clustring(
                    preprocessed_train_set
                )

                labelled_test_set = cluster_creator.initilize_clustring(
                    preprocessed_test_set
                )

                X_train = labelled_train_set.drop(columns=[TARGET_COLUMN], axis=1)
                y_train = labelled_train_set[TARGET_COLUMN]

                X_test = labelled_test_set.drop(columns=[TARGET_COLUMN], axis=1)
                y_test = labelled_test_set[TARGET_COLUMN]

                train_arr = np.c_[np.array(X_train), np.array(y_train)]

                test_arr = np.c_[np.array(X_test, y_test)]

                self.utils.save_numpy_array_data(
                    self.dataTransformationConfig.transformed_train_file_path
                )
                self.utils.save_numpy_array_data(
                    self.dataTransformationConfig.transformed_test_file_path
                )

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.dataTransformationConfig.transformed_object_file_path,
                    transformed_train_file_path=self.dataTransformationConfig.transformed_train_file_path,
                    transformed_test_file_path=self.dataTransformationConfig.transformed_test_file_path,
                )

                logging.info(
                    "Exited initiate_data_transformation method of DataTransformation class"
                )
                return data_transformation_artifact
            else:
                raise Exception("Data validation Failed.")

        except Exception as e:
            raise CustomException(e, sys)
