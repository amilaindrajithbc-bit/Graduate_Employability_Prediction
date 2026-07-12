import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from employability_prediction.logger import logger
from employability_prediction.exception import CustomException
from employability_prediction.utils.helper import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "scaler.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # ===========================================================
    # Feature Engineering
    # ===========================================================

    def feature_engineering(self, df):

        logger.info("Feature Engineering Started")

        feature_df = df.copy()

        # -------------------------------------------
        # Academic Performance
        # -------------------------------------------

        feature_df["academic_performance"] = (
            feature_df["cgpa"] -
            (feature_df["backlogs"] * 0.5)
        )

        # -------------------------------------------
        # Technical Skill Score
        # -------------------------------------------

        feature_df["technical_skill_score"] = (
            feature_df["coding_skills"] +
            feature_df["dsa_score"] +
            feature_df["ml_knowledge"] +
            feature_df["system_design"]
        ) / 4

        # -------------------------------------------
        # Soft Skill Score
        # -------------------------------------------

        feature_df["soft_skill_score"] = (
            feature_df["communication_skills"] +
            feature_df["aptitude_score"] / 10
        ) / 2

        # -------------------------------------------
        # Experience Score
        # -------------------------------------------

        feature_df["experience_score"] = (
            feature_df["internships"] * 3 +
            feature_df["projects_count"] * 2 +
            feature_df["open_source_contributions"]
        )

        # -------------------------------------------
        # Profile Strength
        # -------------------------------------------

        feature_df["profile_strength"] = (
            feature_df["certifications"] +
            feature_df["hackathons"] +
            feature_df["extracurriculars"]
        )

        # -------------------------------------------
        # Employability Score
        # -------------------------------------------

        feature_df["employability_score"] = (
            0.30 * feature_df["academic_performance"] +
            0.30 * feature_df["technical_skill_score"] +
            0.20 * feature_df["experience_score"] +
            0.20 * feature_df["profile_strength"]
        )

        # -------------------------------------------
        # Career Readiness
        # -------------------------------------------

        feature_df["career_readiness"] = (
            feature_df["technical_skill_score"] +
            feature_df["soft_skill_score"] +
            feature_df["experience_score"]
        ) / 3

        # -------------------------------------------
        # Overall Student Performance
        # -------------------------------------------

        feature_df["overall_student_performance"] = (
            feature_df["academic_performance"] +
            feature_df["career_readiness"]
        ) / 2

        logger.info("Feature Engineering Completed")

        return feature_df

    # ===========================================================
    # Preprocessor Object
    # ===========================================================

    def get_data_transformer_object(self):

        try:

            logger.info("Creating Preprocessing Object")

            numerical_columns = [

                "cgpa",
                "backlogs",
                "coding_skills",
                "dsa_score",
                "aptitude_score",
                "communication_skills",
                "ml_knowledge",
                "system_design",
                "internships",
                "projects_count",
                "certifications",
                "hackathons",
                "open_source_contributions",
                "extracurriculars",

                # Engineered Features

                "academic_performance",
                "technical_skill_score",
                "soft_skill_score",
                "experience_score",
                "profile_strength",
                "employability_score",
                "career_readiness",
                "overall_student_performance"

            ]

            categorical_columns = [

                "branch",
                "college_tier"

            ]

            numerical_pipeline = Pipeline(

                steps=[

                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )

                ]

            )

            categorical_pipeline = Pipeline(

                steps=[

                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),

                    (
                        "onehotencoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )

                ]

            )

            preprocessor = ColumnTransformer(

                [

                    (
                        "num_pipeline",
                        numerical_pipeline,
                        numerical_columns
                    ),

                    (
                        "cat_pipeline",
                        categorical_pipeline,
                        categorical_columns
                    )

                ]

            )

            logger.info("Preprocessor Created Successfully")

            return preprocessor

        except Exception as e:

            raise CustomException(e, sys)

    # ===========================================================
    # Data Transformation
    # ===========================================================

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:

            logger.info("Reading Train and Test Dataset")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Train and Test Dataset Loaded Successfully")

            # ===================================================
            # Feature Engineering
            # ===================================================

            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            logger.info("Feature Engineering Applied Successfully")

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "placement_status"

            input_feature_train_df = train_df.drop(
                columns=[target_column]
                
            )

            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(
                columns=[target_column]
                
            )

            target_feature_test_df = test_df[target_column]

            logger.info("Applying Preprocessing Object")

          
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()  
            train_arr = np.c_[

                input_feature_train_arr,
                np.array(target_feature_train_df)

            ]
            test_arr = np.c_[

                input_feature_test_arr,
                np.array(target_feature_test_df)

            ]      

            os.makedirs(
                "artifacts",
                exist_ok=True
            )

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,

                obj=preprocessing_obj

            )

            logger.info("Preprocessor Saved Successfully")

            logger.info("Data Transformation Completed")

            return (

                train_arr,

                test_arr,

                self.data_transformation_config.preprocessor_obj_file_path

            )

        except Exception as e:

            raise CustomException(e, sys)    