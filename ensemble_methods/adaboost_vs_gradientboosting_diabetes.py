"""
adaboost_gradientboosting_diabetes.py
------------------------------------

Purpose:
- Practice ensemble learning methods using scikit-learn
- Compare AdaBoost and Gradient Boosting classifiers
- Perform hyperparameter tuning using GridSearchCV
- Evaluate models using Accuracy, Sensitivity (Recall), and Specificity

Dataset:
- Pima Indians Diabetes Dataset (diabetes.csv)

Note:
This script is intended for learning and experimentation.
It is not a production or deployment-ready pipeline.
"""

# ===============================
# 1. Imports
# ===============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

# ===============================
# 2. Load Dataset
# ===============================

data = pd.read_csv("data/diabetes.csv")

# Split features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ===============================
# 3. AdaBoost + GridSearchCV
# ===============================

base_model = DecisionTreeClassifier(random_state=42)

ada_model = AdaBoostClassifier(
    base_estimator=base_model,
    random_state=42
)

param_grid = {
    "n_estimators": [50, 100],
    "learning_rate": [0.3, 0.6, 0.9]
}

ada_search = GridSearchCV(
    estimator=ada_model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

ada_search.fit(X_train, y_train)

print("\nBest AdaBoost Parameters:")
print(ada_search.best_params_)

# Train AdaBoost with best parameters
best_ada = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=ada_search.best_params_["n_estimators"],
    learning_rate=ada_search.best_params_["learning_rate"],
    random_state=42
)

best_ada.fit(X_train, y_train)
y_pred_ada = best_ada.predict(X_test)

# ===============================
# 4. AdaBoost Evaluation
# ===============================

cm_ada = confusion_matrix(y_test, y_pred_ada)

accuracy_ada = accuracy_score(y_test, y_pred_ada)
sensitivity_ada = recall_score(y_test, y_pred_ada)          # Recall for class 1
specificity_ada = recall_score(y_test, y_pred_ada, pos_label=0)

print("\nAdaBoost Confusion Matrix:\n", cm_ada)
print("AdaBoost Accuracy    :", round(accuracy_ada, 2))
print("AdaBoost Sensitivity :", round(sensitivity_ada, 2))
print("AdaBoost Specificity :", round(specificity_ada, 2))

# ===============================
# 5. Gradient Boosting
# ===============================

gb_model = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.3,
    random_state=42
)

gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# ===============================
# 6. Gradient Boosting Evaluation
# ===============================

cm_gb = confusion_matrix(y_test, y_pred_gb)

accuracy_gb = accuracy_score(y_test, y_pred_gb)
sensitivity_gb = recall_score(y_test, y_pred_gb)
specificity_gb = recall_score(y_test, y_pred_gb, pos_label=0)

print("\nGradient Boosting Confusion Matrix:\n", cm_gb)
print("Gradient Boosting Accuracy    :", round(accuracy_gb, 2))
print("Gradient Boosting Sensitivity :", round(sensitivity_gb, 2))
print("Gradient Boosting Specificity :", round(specificity_gb, 2))

# ===============================
# 7. Results Summary
# ===============================

results = pd.DataFrame({
    "Model": ["AdaBoost", "Gradient Boosting"],
    "Accuracy": [accuracy_ada, accuracy_gb],
    "Sensitivity": [sensitivity_ada, sensitivity_gb],
    "Specificity": [specificity_ada, specificity_gb]
})

print("\nModel Comparison Summary:\n")
print(results.round(2))

# Optional: save results
results.to_csv("ensemble_results_diabetes.csv", index=False)
