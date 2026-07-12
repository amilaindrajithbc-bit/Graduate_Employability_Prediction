import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st

from employability_prediction.pipeline.prediction_pipeline import (
    PredictionPipeline,
    CustomData
)

# ===========================================================
# Page Configuration
# ===========================================================

st.set_page_config(
    page_title="Graduate Employability Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================================
# Sidebar
# ===========================================================

with st.sidebar:

    st.title("🎓 Graduate Employability")

    st.markdown("---")

    st.markdown("## Project Information")

    st.write("🎓 MSc Computer Science")

    st.write("🤖 Machine Learning Project")

    st.write("📈 Model : Logistic Regression")

    st.write("🎯 Accuracy : 94.21%")

    st.markdown("---")

    st.info(
        """
        Fill in the student's academic,
        technical and experience details,
        then click **Predict Placement**.
        """
    )

# ===========================================================
# Main Title
# ===========================================================

st.title("🎓 Graduate Employability Prediction System")

st.markdown(
"""
### Predict Whether a Graduate is Likely to be Placed

This application predicts employability using a Machine Learning model based on:

- 📚 Academic Performance
- 💻 Technical Skills
- 🏆 Experience & Achievements
- 🎯 Employability Indicators
"""
)

st.divider()

# ===========================================================
# Student Input Form
# ===========================================================

st.header("📝 Enter Student Details")

tab1, tab2, tab3 = st.tabs(
    [
        "📚 Academic",
        "💻 Technical Skills",
        "🏆 Experience"
    ]
)

# ===========================================================
# Academic Information
# ===========================================================

with tab1:

    st.subheader("Academic Information")

    col1, col2 = st.columns(2)

    with col1:

        branch = st.selectbox(
            "Branch",
            [
                "CSE",
                "IT",
                "ECE",
                "EE",
                "Mechanical",
                "Chemical"
            ]
        )

        cgpa = st.number_input(
            "CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.5,
            step=0.1
        )

        aptitude_score = st.slider(
            "Aptitude Score",
            min_value=0,
            max_value=100,
            value=70
        )

    with col2:

        college_tier = st.selectbox(
            "College Tier",
            [
                "Tier-1",
                "Tier-2",
                "Tier-3"
            ]
        )

        backlogs = st.number_input(
            "Backlogs",
            min_value=0,
            max_value=20,
            value=0
        )

        communication_skills = st.slider(
            "Communication Skills",
            min_value=0,
            max_value=100,
            value=70
        )

# ===========================================================
# Technical Skills
# ===========================================================

with tab2:

    st.subheader("Technical Skills")

    col1, col2 = st.columns(2)

    with col1:

        coding_skills = st.slider(
            "Coding Skills",
            min_value=0,
            max_value=100,
            value=70
        )

        ml_knowledge = st.slider(
            "Machine Learning Knowledge",
            min_value=0,
            max_value=100,
            value=60
        )

    with col2:

        dsa_score = st.slider(
            "DSA Score",
            min_value=0,
            max_value=100,
            value=70
        )

        system_design = st.slider(
            "System Design",
            min_value=0,
            max_value=100,
            value=60
        )

# ===========================================================
# Experience & Achievements
# ===========================================================

with tab3:

    st.subheader("Experience & Achievements")

    col1, col2 = st.columns(2)

    with col1:

        internships = st.number_input(
            "Internships",
            min_value=0,
            max_value=20,
            value=1
        )

        projects_count = st.number_input(
            "Projects",
            min_value=0,
            max_value=50,
            value=3
        )

        certifications = st.number_input(
            "Certifications",
            min_value=0,
            max_value=50,
            value=2
        )

    with col2:

        hackathons = st.number_input(
            "Hackathons",
            min_value=0,
            max_value=50,
            value=1
        )

        open_source_contributions = st.number_input(
            "Open Source Contributions",
            min_value=0,
            max_value=100,
            value=0
        )

        extracurriculars = st.number_input(
            "Extracurricular Activities",
            min_value=0,
            max_value=50,
            value=1
        )

st.divider()
# ===========================================================
# Prediction Button
# ===========================================================

if st.button("🚀 Predict Placement", use_container_width=True):

    try:

        student_data = CustomData(

            branch=branch,

            college_tier=college_tier,

            cgpa=cgpa,

            backlogs=backlogs,

            coding_skills=coding_skills,

            dsa_score=dsa_score,

            aptitude_score=aptitude_score,

            communication_skills=communication_skills,

            ml_knowledge=ml_knowledge,

            system_design=system_design,

            internships=internships,

            projects_count=projects_count,

            certifications=certifications,

            hackathons=hackathons,

            open_source_contributions=open_source_contributions,

            extracurriculars=extracurriculars

        )

        pred_df = student_data.get_data_as_dataframe()

        predict_pipeline = PredictionPipeline()

        prediction = predict_pipeline.predict(pred_df)

        st.divider()

        st.header("📊 Prediction Result")

        col1, col2 = st.columns([2, 1])

        with col1:

            if prediction[0] == 1:

                st.success(
                    """
### 🎉 Placement Prediction : PLACED

The student is likely to be placed.

This prediction is based on academic performance,
technical skills, and experience.
"""
                )

            else:

                st.error(
                    """
### ❌ Placement Prediction : NOT PLACED

The student is unlikely to be placed.

Improving technical skills, academic performance,
and practical experience may improve employability.
"""
                )

        with col2:

            st.metric(

                label="Model",

                value="Logistic Regression"

            )

            st.metric(

                label="Accuracy",

                value="94.21%"

            )

            st.metric(

                label="Prediction",

                value="Placed" if prediction[0] == 1 else "Not Placed"

            )

        # ===========================================================
        # Student Summary
        # ===========================================================

        st.divider()

        st.subheader("📋 Student Summary")

        summary_df = pred_df[
            [
                "branch",
                "college_tier",
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
                "extracurriculars"
            ]
        ]

        st.dataframe(

            summary_df,

            use_container_width=True

        )

    except Exception as e:

        st.error(f"Error : {e}")


# ===========================================================
# Footer
# ===========================================================

st.divider()

st.caption(

"""
Graduate Employability Prediction System

Developed for an MSc Computer Science Machine Learning Project

Model:
Logistic Regression

Training Accuracy:
94.21%

Built with:
Python • Scikit-learn • Streamlit
"""

)
