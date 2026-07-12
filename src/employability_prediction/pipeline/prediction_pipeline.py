import os
import sys

import pandas as pd

from employability_prediction.exception import CustomException
from employability_prediction.utils.helper import load_object


# ===========================================================
# Prediction Pipeline
# ===========================================================

class PredictionPipeline:

    def __init__(self):

        self.model_path = os.path.join(
            "artifacts",
            "model.pkl"
        )

        self.preprocessor_path = os.path.join(
            "artifacts",
            "scaler.pkl"
        )

    def predict(self, features):

        try:

            model = load_object(self.model_path)

            preprocessor = load_object(
                self.preprocessor_path
            )

            data_scaled = preprocessor.transform(
                features
            )

            prediction = model.predict(
                data_scaled
            )

            return prediction

        except Exception as e:

            raise CustomException(e, sys)


# ===========================================================
# Custom Input Data
# ===========================================================

class CustomData:

    def __init__(

        self,

        branch,
        college_tier,
        cgpa,
        backlogs,
        coding_skills,
        dsa_score,
        aptitude_score,
        communication_skills,
        ml_knowledge,
        system_design,
        internships,
        projects_count,
        certifications,
        hackathons,
        open_source_contributions,
        extracurriculars

    ):

        self.branch = branch
        self.college_tier = college_tier
        self.cgpa = cgpa
        self.backlogs = backlogs
        self.coding_skills = coding_skills
        self.dsa_score = dsa_score
        self.aptitude_score = aptitude_score
        self.communication_skills = communication_skills
        self.ml_knowledge = ml_knowledge
        self.system_design = system_design
        self.internships = internships
        self.projects_count = projects_count
        self.certifications = certifications
        self.hackathons = hackathons
        self.open_source_contributions = open_source_contributions
        self.extracurriculars = extracurriculars

    # =======================================================
    # Convert User Input to DataFrame
    # =======================================================

    def get_data_as_dataframe(self):

        try:

            custom_data_input_dict = {

                "branch": [self.branch],

                "college_tier": [self.college_tier],

                "cgpa": [self.cgpa],

                "backlogs": [self.backlogs],

                "coding_skills": [self.coding_skills],

                "dsa_score": [self.dsa_score],

                "aptitude_score": [self.aptitude_score],

                "communication_skills": [
                    self.communication_skills
                ],

                "ml_knowledge": [self.ml_knowledge],

                "system_design": [self.system_design],

                "internships": [self.internships],

                "projects_count": [self.projects_count],

                "certifications": [self.certifications],

                "hackathons": [self.hackathons],

                "open_source_contributions": [
                    self.open_source_contributions
                ],

                "extracurriculars": [
                    self.extracurriculars
                ]

            }

            df = pd.DataFrame(custom_data_input_dict)

            # ===================================================
            # Feature Engineering
            # ===================================================

            df["academic_performance"] = (

                df["cgpa"]

                -

                (df["backlogs"] * 0.5)

            )

            df["technical_skill_score"] = (

                df["coding_skills"]

                +

                df["dsa_score"]

                +

                df["ml_knowledge"]

                +

                df["system_design"]

            ) / 4

            df["soft_skill_score"] = (

                df["communication_skills"]

                +

                df["aptitude_score"] / 10

            ) / 2

            df["experience_score"] = (

                df["internships"] * 3

                +

                df["projects_count"] * 2

                +

                df["open_source_contributions"]

            )

            df["profile_strength"] = (

                df["certifications"]

                +

                df["hackathons"]

                +

                df["extracurriculars"]

            )

            df["employability_score"] = (

                0.30 * df["academic_performance"]

                +

                0.30 * df["technical_skill_score"]

                +

                0.20 * df["experience_score"]

                +

                0.20 * df["profile_strength"]

            )

            df["career_readiness"] = (

                df["technical_skill_score"]

                +

                df["soft_skill_score"]

                +

                df["experience_score"]

            ) / 3

            df["overall_student_performance"] = (

                df["academic_performance"]

                +

                df["career_readiness"]

            ) / 2

            return df

        except Exception as e:

            raise CustomException(e, sys)