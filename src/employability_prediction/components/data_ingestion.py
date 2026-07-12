import os
import sys

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from employability_prediction.logger import logger
from employability_prediction.exception import CustomException


@dataclass
class DataIngestionConfig:

    raw_data_path = os.path.join(
        "data",
        "raw",
        "student_placement_synthetic.csv"
    )

    train_data_path = os.path.join(
        "data",
        "processed",
        "train.csv"
    )

    test_data_path = os.path.join(
        "data",
        "processed",
        "test.csv"
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logger.info("Data Ingestion Started")

        try:

            # Read Dataset
            df = pd.read_csv(
                self.ingestion_config.raw_data_path
            )

            logger.info("Dataset Loaded Successfully")

            # Create Processed Folder
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

            # Train-Test Split
            train_set, test_set = train_test_split(
                df,
                test_size=0.20,
                random_state=42,
                stratify=df["placement_status"]
            )

            logger.info("Train-Test Split Completed")

            # Save Train Dataset
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            logger.info("train.csv Saved Successfully")

            # Save Test Dataset
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logger.info("test.csv Saved Successfully")

            logger.info("Data Ingestion Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:

            raise CustomException(e, sys)