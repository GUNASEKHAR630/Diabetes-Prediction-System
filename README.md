# Diabetes Prediction System

This project predicts whether a person is likely to have diabetes using Machine Learning techniques in Python. The project demonstrates a complete end-to-end Machine Learning workflow including data preprocessing, feature engineering, model training, hyperparameter tuning, and model evaluation.

## Features
- Data Cleaning and Preprocessing
- Handling Missing Values
- Feature Scaling
- Exploratory Data Analysis (EDA)
- Train-Test Split
- Machine Learning Model Training
- Hyperparameter Tuning using GridSearchCV
- Diabetes Prediction
- Model Evaluation and Visualization

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Machine Learning Concepts Used
- Pipeline
- ColumnTransformer
- SimpleImputer
- StandardScaler
- RandomForestClassifier
- LogisticRegression
- GridSearchCV
- Confusion Matrix
- Classification Report

## Dataset
The dataset contains patient-related medical information such as:
- Glucose Level
- Blood Pressure
- BMI
- Insulin Level
- Age
- Skin Thickness
- Pregnancies

Target Variable:
- Diabetes Outcome (0 = No Diabetes, 1 = Diabetes)

## Project Workflow
1. Load Dataset
2. Perform Data Cleaning
3. Handle Missing Values
4. Perform Exploratory Data Analysis
5. Scale Numerical Features
6. Split Dataset into Training and Testing Data
7. Train Machine Learning Model
8. Tune Hyperparameters using GridSearchCV
9. Evaluate Model Performance
10. Predict Diabetes Outcome

## Output
The model predicts whether a patient is diabetic and displays:
- Accuracy Score
- Confusion Matrix
- Classification Report
- Best Hyperparameters

## How to Run

Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
