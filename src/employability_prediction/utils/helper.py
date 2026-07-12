import os
import sys
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.model_selection import GridSearchCV

from employability_prediction.logger import logger
from employability_prediction.exception import CustomException


# ===========================================================
# Save Object
# ===========================================================

def save_object(file_path, obj):

    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:

            pickle.dump(obj, file_obj)

        logger.info(f"Object saved successfully at {file_path}")

    except Exception as e:

        raise CustomException(e, sys)


# ===========================================================
# Load Object
# ===========================================================

def load_object(file_path):

    try:

        with open(file_path, "rb") as file_obj:

            logger.info(f"Object loaded successfully from {file_path}")

            return pickle.load(file_obj)

    except Exception as e:

        raise CustomException(e, sys)


# ===========================================================
# Evaluate Models
# ===========================================================

def evaluate_models(

    X_train,
    y_train,
    X_test,
    y_test,
    models,
    params

):

    try:

        report = {}

        for model_name in models:

            model = models[model_name]

            param = params[model_name]

            gs = GridSearchCV(

                estimator=model,

                param_grid=param,

                cv=5,

                n_jobs=-1,

                verbose=0

            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)

            report[model_name] = {

                "best_model": best_model,

                "accuracy": accuracy_score(
                    y_test,
                    y_pred
                ),

                "precision": precision_score(
                    y_test,
                    y_pred
                ),

                "recall": recall_score(
                    y_test,
                    y_pred
                ),

                "f1_score": f1_score(
                    y_test,
                    y_pred
                ),

                "roc_auc": roc_auc_score(
                    y_test,
                    y_pred
                )

            }

        return report

    except Exception as e:

        raise CustomException(e, sys)