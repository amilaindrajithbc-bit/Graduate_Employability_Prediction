import os
import sys

import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

from employability_prediction.logger import logger
from employability_prediction.exception import CustomException
from employability_prediction.utils.helper import (
    evaluate_models,
    save_object
)


@dataclass
class ModelTrainerConfig:

    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logger.info("Splitting Training and Testing Data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logger.info("Training and Testing Data Split Successfully")

            # ==================================================
            # Models
            # ==================================================

            models = {

                "Logistic Regression": LogisticRegression(),

                "Random Forest": RandomForestClassifier(),

                "Extra Trees": ExtraTreesClassifier(),

                "XGBoost": XGBClassifier(
                    eval_metric="logloss"
                )

            }

            # ==================================================
            # Hyperparameter Grid
            # ==================================================

            params = {

                "Logistic Regression": {

                    "C": [0.1, 1, 10],

                    "solver": ["lbfgs"],

                    "max_iter": [1000]

                },

                "Random Forest": {

                    "n_estimators": [100, 200],

                    "max_depth": [10, 20, None],

                    "min_samples_split": [2, 5]

                },

                "Extra Trees": {

                    "n_estimators": [100, 200],

                    "max_depth": [10, 20, None]

                },

                "XGBoost": {

                    "n_estimators": [100, 200],

                    "learning_rate": [0.05, 0.1],

                    "max_depth": [3, 5]

                }

            }

            logger.info("Model Training Started")

            model_report = evaluate_models(

                X_train,

                y_train,

                X_test,

                y_test,

                models,

                params

            )

            logger.info("Model Evaluation Completed")

            # ==================================================
            # Select Best Model
            # ==================================================

            best_model_name = max(
                model_report,
                key=lambda x: model_report[x]["accuracy"]
            )

            best_model = model_report[best_model_name]["best_model"]

            best_model_score = model_report[best_model_name]["accuracy"]

            logger.info(
                f"Best Model : {best_model_name}"
            )

            logger.info(
                f"Best Accuracy : {best_model_score}"
            )

            # ==================================================
            # Save Best Model
            # ==================================================

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model

            )

            logger.info("Best Model Saved Successfully")

            # ==================================================
            # Print Results
            # ==================================================

            print("\n==============================")
            print("Model Performance")
            print("==============================")

            for model_name, result in model_report.items():

                print(f"\n{model_name}")

                print(f"Accuracy  : {result['accuracy']:.4f}")
                print(f"Precision : {result['precision']:.4f}")
                print(f"Recall    : {result['recall']:.4f}")
                print(f"F1 Score  : {result['f1_score']:.4f}")
                print(f"ROC AUC   : {result['roc_auc']:.4f}")

            print("\n==============================")
            print(f"Best Model : {best_model_name}")
            print(f"Accuracy   : {best_model_score:.4f}")
            print("==============================")

            return best_model_score

        except Exception as e:

            raise CustomException(e, sys)