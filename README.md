# Mental Health & Stress Analysis Project
This project demonstrates data analysis, visualization, 
and machine learning using a real mental health survey dataset.

## Project Structure
## Project Goals
* Data loading and overview
* Data cleaning and label encoding
* Exploratory data analysis (EDA)
* Correlation analysis
* Building and comparing multiple machine learning models
* Handling class imbalance with SMOTE
* Evaluating model performance

## Visualizations
* Growing Stress distribution
* Stress by Age Group (countplot)
* Stress by Gender (countplot)
* Stress by Occupation (countplot)
* Stress by Mental Health History (countplot)
* Stress by Mood Swings (countplot)
* Correlation heatmap
* Feature correlation with Growing Stress (barplot)
* Confusion Matrix

## Key Findings
* Best model: Logistic Regression + SMOTE (accuracy 52%)
* Dataset has weak correlations (< 0.1) — stress is hard to predict
* Most variable features: Days_Indoors, Occupation
* SMOTE improved minority class recall from 0.00 to 0.50
* All models compared: RandomForest, RandomForest + SMOTE,
  Logistic Regression, Logistic Regression + SMOTE

## Technologies
* Python
* pandas
* numpy
* seaborn
* matplotlib
* scikit-learn
* imbalanced-learn
* Jupyter Notebook

## Author
vaslinx
