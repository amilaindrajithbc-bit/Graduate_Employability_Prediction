# 🎓 Graduate Employability Prediction System

## 📖 Project Overview

The **Graduate Employability Prediction System** is an end-to-end Machine Learning application developed as part of an **MSc Computer Science** project. The objective of this project is to predict whether a graduate is likely to be placed based on academic performance, technical skills, communication skills, and practical experience.

The project demonstrates the complete Machine Learning lifecycle, including data preprocessing, feature engineering, model training, hyperparameter tuning, model evaluation, model deployment, and web application development using **Streamlit**.

This project follows a modular software engineering approach, where each stage of the Machine Learning pipeline is implemented as an independent Python module, making the application scalable, reusable, and easy to maintain.

---

# 🎯 Project Objectives

The main objectives of this project are:

- Predict graduate employability using Machine Learning.
- Perform data preprocessing and feature engineering.
- Compare multiple Machine Learning algorithms.
- Improve model performance using hyperparameter tuning.
- Build a modular Machine Learning pipeline.
- Deploy the trained model as a Streamlit web application.
- Provide an easy-to-use interface for predicting graduate employability.

---

# 📊 Dataset

### Dataset Name

**Student Placement Synthetic Dataset**

The dataset contains information related to graduate academic performance, technical skills, communication skills, and practical experience.

### Input Features

- Branch
- College Tier
- CGPA
- Backlogs
- Coding Skills
- DSA Score
- Aptitude Score
- Communication Skills
- Machine Learning Knowledge
- System Design
- Internships
- Projects Count
- Certifications
- Hackathons
- Open Source Contributions
- Extracurricular Activities

### Target Variable

- Placement Status

---

# ⚙️ Feature Engineering

Several new features were created to improve the predictive performance of the Machine Learning models.

The engineered features include:

- Academic Performance
- Technical Skill Score
- Soft Skill Score
- Experience Score
- Profile Strength
- Employability Score
- Career Readiness
- Overall Student Performance

These engineered features significantly improved the predictive capability of the final model.

---

# 🤖 Machine Learning Models

The following Machine Learning algorithms were implemented and evaluated:

- Logistic Regression
- Random Forest Classifier
- Extra Trees Classifier
- XGBoost Classifier

Hyperparameter tuning was performed using **GridSearchCV** to obtain the optimal model parameters.

---

# 🏆 Best Performing Model

After evaluating all models, the best-performing model was:

**Logistic Regression**

### Final Accuracy

**94.21%**

The trained model is stored as:

```text
artifacts/model.pkl
```

The preprocessing pipeline is stored as:

```text
artifacts/scaler.pkl
```
# 📁 Project Structure

```text
Graduate_Employability_Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── setup.py
├── .gitignore
├── template.py
│
├── artifacts/
│   ├── model.pkl
│   └── scaler.pkl
│
├── data/
│   ├── raw/
│   │   └── student_placement_synthetic.csv
│   │
│   └── processed/
│       ├── train.csv
│       └── test.csv
│
├── notebooks/
│   └── employability_prediction.ipynb
│
├── logs/
│
├── src/
│   └── employability_prediction/
│       │
│       ├── components/
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   └── model_trainer.py
│       │
│       ├── pipeline/
│       │   ├── train_pipeline.py
│       │   └── prediction_pipeline.py
│       │
│       ├── utils/
│       │   └── helper.py
│       │
│       ├── logger.py
│       └── exception.py
│
└── venv/
```

---

# 💻 Development Environment

This project was developed and tested using the following environment:

- Visual Studio Code
- Visual Studio Code Integrated Terminal
- Python 3.11
- Windows Operating System
- Jupyter Notebook
- Streamlit

---

# 📋 Prerequisites

Before running this project, ensure the following software is installed:

- Python 3.11
- Visual Studio Code
- Git
- Jupyter Notebook

---
# ⚙️ Installation

Open the **Visual Studio Code Integrated Terminal**.

### Step 1 – Create a Conda Environment

```cmd
conda create -p venv python=3.11 -y
```

### Step 2 – Activate the Conda Environment

```cmd
conda activate ./venv
```

### Step 3 – Install Required Packages

```cmd
pip install -r requirements.txt
```

This installs all the required Python packages needed to run the Jupyter Notebook, Machine Learning pipeline, and Streamlit application.

# ▶️ Running the Jupyter Notebook

open:

```text
notebooks\employability_prediction.ipynb
```

Run all notebook cells sequentially to reproduce the complete Machine Learning workflow.

# 🌐 Running the Streamlit Application

Open the **Visual Studio Code Integrated Terminal** and execute:

```cmd
streamlit run app.py

The application will automatically open in your default web browser.

---

# 📈 Model Performance

The Machine Learning models were evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.



| Logistic Regression | **94.21%** | 94.35% | 95.06% | 94.70% | 94.13% |
| Random Forest | 91.09% | 91.33% | 92.40% | 91.86% | 90.96% |
| Extra Trees | 91.33% | 91.41% | 92.80% | 92.10% | 91.19% |
| XGBoost | 92.58% | 92.74% | 93.71% | 93.22% | 92.47% |

After comparing all models, **Logistic Regression** achieved the highest performance and was selected as the final model for deployment.

---

# 🚀 Application Features

The Streamlit application provides the following features:

- Professional user interface
- Student information input form
- Academic information tab
- Technical skills tab
- Experience and achievements tab
- Automatic feature engineering
- Machine Learning prediction
- Student summary
- Placement prediction
- Responsive web interface

---

# 🛠️ Technologies Used

The following technologies were used to develop this project:

### Programming Language

- Python 3.11

### Development Environment

- Visual Studio Code
- Visual Studio Code Integrated Terminal
- Jupyter Notebook

### Python Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Plotly
- SciPy

---

# 📸 Project Screenshots

The project includes screenshots demonstrating:

- Dataset loading
- Data preprocessing
- Feature engineering
- Exploratory Data Analysis (EDA)
- Model training
- Hyperparameter tuning
- Model evaluation
- Streamlit application
- Placement prediction results

---

# 📂 Repository Contents

This repository contains:

- Jupyter Notebook implementation
- Machine Learning pipeline
- Data preprocessing modules
- Feature engineering modules
- Model training modules
- Prediction pipeline
- Streamlit web application
- Trained Machine Learning model
- Preprocessing pipeline
- Project documentation

---

# 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

### GitHub Repository

**    **

### Streamlit Live Application

**   **

---

# 🔮 Future Improvements

Future enhancements may include:

- Resume analysis using NLP
- Interview performance prediction
- Deep Learning models
- Explainable AI (SHAP/LIME)
- User authentication
- Database integration
- Cloud deployment with CI/CD
- Mobile-friendly interface

---

# 👨‍💻 Author

**Amila Piyasiri**

MSc Computer Science

Graduate Employability Prediction System

Developed using Python, Scikit-learn, XGBoost, and Streamlit.

---

# 📄 License

This project has been developed for **academic purposes** as part of an MSc Computer Science programme.

