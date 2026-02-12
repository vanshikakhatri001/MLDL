# ===============================
# SVM Classification with Hyperparameter Tuning
# ===============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ------------------------------
# 1) Load Dataset
# ------------------------------
# Using scikit-learn built-in Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("Features:", feature_names)
print("Classes:", target_names)

# ------------------------------
# 2) Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# 3) Baseline SVM Model
# ------------------------------
svm = SVC()
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print("\nBaseline Accuracy:", accuracy_score(y_test, y_pred))

# ------------------------------
# 4) Hyperparameter Tuning with GridSearchCV
# ------------------------------
param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf", "poly"],
    "gamma": ["scale", "auto"],
    "degree": [2, 3, 4]
}

grid = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    verbose=2,
    n_jobs=1   
)

grid.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid.best_params_)
print("Best Cross-Validation Score:", grid.best_score_)

best_model = grid.best_estimator_

# ------------------------------
# 5) Final Evaluation
# ------------------------------
y_pred_best = best_model.predict(X_test)

print("\nFinal Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
