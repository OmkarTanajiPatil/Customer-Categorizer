import sys
import os


from src.logger import logging
from src.exception import CustomException

from src.constants.training_pipeline import *

from src.entity.config_entity import PCAConfig, ClusteringConfig

from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class CreateCluster:
    def __init__(self):
        self.pca_config = PCAConfig()
        self.clustering_config = ClusteringConfig()

    def get_dataset_using_pca(self, preprocessed_data: DataFrame) -> DataFrame:
        try:
            pca_object = PCA(**self.pca_config.get_pca_config).fit(preprocessed_data)

            reduced_dataset = pca_object.fit_transform(preprocessed_data)

            logging.info("PCA transformation is done")
            return reduced_dataset

        except Exception as e:
            raise CustomException(e, sys)

    def initilize_clustring(self, preprocessed_data: DataFrame) -> DataFrame:
        try:
            logging.info("Initilizing clustering...")

            reduced_dataset = self.get_dataset_using_pca(preprocessed_data)

            model = KMeans(n_clusters=self.clustering_config.n_clusters).fit(
                reduced_dataset
            )

            preprocessed_data[TARGET_COLUMN] = model.labels_.astype(int)

            logging.info("Clustering is Done!")

            return preprocessed_data
        except Exception as e:
            raise CustomException(e, sys)
