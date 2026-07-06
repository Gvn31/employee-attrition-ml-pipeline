# employee-attrition-mlops-pipeline

# Project Overview

This project aims to develop an end-to-end Machine Learning Operations (MLOps) pipeline to predict employee attrition using employee demographic and job-related information. The pipeline covers data analysis, preprocessing, feature engineering, model training, evaluation, and deployment.


# Phase 1: Problem Definition

## 1. Business Goal

Employee attrition is a major challenge for organizations as it leads to increased recruitment costs, productivity loss, and knowledge gaps. The goal of this project is to build a machine learning model that predicts whether an employee is likely to leave the organization. This enables the Human Resources (HR) department to identify at-risk employees and take proactive retention measures.

## 2. Machine Learning Task
Problem Type: Binary Classification

Target Variable: Attrition

Classes:
    - Stayed
    - Left

The model predicts whether an employee will stay with the organization or leave.


## 3. Feasibility Assessment

- The employee attrition dataset is available.
- The dataset contains sufficient employee demographic and job-related features.
- Binary classification is suitable for the problem.
- Model predictions should be interpretable for HR decision-making.
- Real-time prediction is not mandatory; batch prediction is sufficient.

## 4. Key Performance Indicators (KPIs)

### Model KPIs

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

### System KPIs

- Low inference time
- Reliable prediction performance
- Maintainable and reproducible ML pipeline


## 5. Baseline Model

**Logistic Regression**
A simple baseline model will be used initially to establish reference performance.


## 6. Final Model

**Random Forest Classifier**

The final model will be selected based on evaluation metrics and overall performance.

---

# Project Workflow
```
Repository Setup
        ↓
Problem Definition
        ↓
Exploratory Data Analysis (EDA)
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Deployment
```


# Project Structure

```
employee-attrition-mlops-pipeline/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│
├── models/
│
├── reports/
│
├── app/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```


# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Git
- GitHub
- Conda


# Current Progress

- [x] Repository Setup
- [x] Virtual Environment Configuration
- [x] Project Structure Creation
- [x] Exploratory Data Analysis (EDA)
- [ ] Data Preprocessing
- [ ] Feature Engineering
- [ ] Model Training
- [ ] Model Evaluation
- [ ] Deployment


# Author
Developed as part of an MLOps learning project for Employee Attrition Prediction.