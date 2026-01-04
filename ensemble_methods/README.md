# Ensemble Methods – AdaBoost vs Gradient Boosting

## Overview

This folder contains a focused machine learning experiment comparing
**AdaBoost** and **Gradient Boosting** classifiers using the
Pima Indians Diabetes dataset.

The purpose of this experiment is to understand:
- How ensemble methods behave on a small tabular dataset
- The impact of hyperparameters such as `n_estimators` and `learning_rate`
- Differences between boosting techniques in terms of sensitivity and specificity

This is a **learning and experimentation script**, not a production pipeline.

---

## Experiment Details

### Dataset
- Pima Indians Diabetes Dataset (`diabetes.csv`)
- Target variable:
  - `0` → No diabetes
  - `1` → Diabetes

---

### Models Used
- **AdaBoostClassifier**
  - Base estimator: Decision Tree
  - Hyperparameter tuning using GridSearchCV
- **GradientBoostingClassifier**
  - Fixed hyperparameters for comparison

---

### Evaluation Metrics
The models are evaluated using:
- **Accuracy**
- **Sensitivity (Recall for positive class)**
- **Specificity (Recall for negative class)**
- Confusion Matrix

These metrics help understand model behavior beyond accuracy alone.

---

## File Description

- `adaboost_gradientboosting_diabetes.py`  
  Contains the full experiment:
  - Data loading
  - Train–test split
  - Hyperparameter tuning for AdaBoost
  - Model training
  - Metric evaluation
  - Result comparison

---

## Notes

- This code is intended for **practice and conceptual understanding**
- No deployment or production integration is included
- Larger, end-to-end ML systems are maintained in separate repositories

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
