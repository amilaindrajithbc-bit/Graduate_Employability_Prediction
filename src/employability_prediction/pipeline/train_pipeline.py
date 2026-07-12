from employability_prediction.components.data_ingestion import DataIngestion
from employability_prediction.components.data_transformation import DataTransformation
from employability_prediction.components.model_trainer import ModelTrainer

from employability_prediction.logger import logger


if __name__ == "__main__":

    logger.info("Training Pipeline Started")

    # ===============================
    # Data Ingestion
    # ===============================

    data_ingestion = DataIngestion()

    train_path, test_path = data_ingestion.initiate_data_ingestion()

    # ===============================
    # Data Transformation
    # ===============================

    data_transformation = DataTransformation()

    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_path,
        test_path
    )

    # ===============================
    # Model Training
    # ===============================

    model_trainer = ModelTrainer()

    model_trainer.initiate_model_trainer(
        train_arr,
        test_arr
    )

    logger.info("Training Pipeline Completed Successfully")