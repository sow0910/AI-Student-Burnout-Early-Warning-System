# AI-Powered Student Burnout & Dropout Early Warning System

## Problem Statement
Early detection of student burnout and dropout risk using behavioural analytics.

## Dataset Type
Synthetic Dataset

## Why Synthetic?
No real behavioural LMS dataset was available. Therefore, behavioural features were simulated using realistic academic behaviour patterns.

## Dataset Description
Number of Records: 1500 students

Features:
- LMS login frequency
- Attendance percentage
- Assignment delay days
- GPA
- Study hours
- Stress score
- Department
- Gender

## Feature Engineering
- One-hot encoding for categorical variables
- Feature scaling for numeric features
- Behavioural risk scoring logic

## Models Used
- Random Forest (Burnout Prediction)
- Logistic Regression (Dropout Probability)

## Outputs
- Burnout Risk Level
- Dropout Probability
- Risk Score (0–100)
- Recommended Intervention Strategy
- Visualization Dashboard

## Tech Stack
- Python
- Scikit-learn
- Streamlit
- Pandas
- Matplotlib
