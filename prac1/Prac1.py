# ===============================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, classification_report
)

# ===============================
# STEP 2: LOAD DATASET
# ===============================
data = pd.read_csv("insurance.csv")

# ===============================
# STEP 3: DATA PREPROCESSING
# ===============================
encoder = LabelEncoder()

data['sex'] = encoder.fit_transform(data['sex'])
data['smoker'] = encoder.fit_transform(data['smoker'])
data['region'] = encoder.fit_transform(data['region'])

# ===============================
# LINEAR REGRESSION MODEL
# ===============================

# Selecting features and target
X_lr = data[['age', 'bmi', 'children', 'smoker', 'region']]
y_lr = data['charges']

# Splitting dataset
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_lr, y_lr, test_size=0.2, random_state=42
)

# Model creation and training
lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)

# Prediction
pred_lr = lr_model.predict(X_test_lr)

# Evaluation metrics
mse_val = mean_squared_error(y_test_lr, pred_lr)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(y_test_lr, pred_lr)

print("\n===== LINEAR REGRESSION OUTPUT =====")
print("MSE :", mse_val)
print("RMSE:", rmse_val)
print("R2 Score:", r2_val)

# Residual Plot
errors = y_test_lr - pred_lr

plt.figure()
plt.scatter(pred_lr, errors)
plt.axhline(y=0)
plt.xlabel("Predicted Values")
plt.ylabel("Residual Errors")
plt.title("Residual Plot for Linear Regression")
plt.show()

# ===============================
# LOGISTIC REGRESSION (WITHOUT TUNING)
# ===============================

# Feature and target
X_log = data[['age', 'bmi', 'children', 'region', 'charges']]
y_log = data['smoker']

# Splitting dataset
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42
)

# Model training
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_log, y_train_log)

# Prediction
pred_log = log_model.predict(X_test_log)

# Evaluation
acc = accuracy_score(y_test_log, pred_log)
conf_mat = confusion_matrix(y_test_log, pred_log)
class_rep = classification_report(y_test_log, pred_log)

print("\n===== LOGISTIC REGRESSION (NO TUNING) =====")
print("Accuracy:", acc)
print("\nConfusion Matrix:\n", conf_mat)
print("\nClassification Report:\n", class_rep)

# Confusion Matrix Plot
plt.figure()
plt.imshow(conf_mat)
plt.title("Confusion Matrix (Without Tuning)")
plt.colorbar()

plt.xticks([0, 1], ["Non-Smoker", "Smoker"])
plt.yticks([0, 1], ["Non-Smoker", "Smoker"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_mat[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# LOGISTIC REGRESSION (WITH TUNING)
# ===============================

# Hyperparameter setup
parameters = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# Grid Search
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000),
    parameters,
    cv=5,
    scoring='accuracy'
)

# Training with best parameters
grid_search.fit(X_train_log, y_train_log)

# Best model
best_model = grid_search.best_estimator_

# Prediction
pred_tuned = best_model.predict(X_test_log)

# Evaluation
acc_tuned = accuracy_score(y_test_log, pred_tuned)
cm_tuned = confusion_matrix(y_test_log, pred_tuned)
rep_tuned = classification_report(y_test_log, pred_tuned)

print("\n===== LOGISTIC REGRESSION (WITH TUNING) =====")
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", acc_tuned)
print("\nConfusion Matrix:\n", cm_tuned)
print("\nClassification Report:\n", rep_tuned)

# Confusion Matrix Plot (Tuned)
plt.figure()
plt.imshow(cm_tuned)
plt.title("Confusion Matrix (With Tuning)")
plt.colorbar()

plt.xticks([0, 1], ["Non-Smoker", "Smoker"])
plt.yticks([0, 1], ["Non-Smoker", "Smoker"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm_tuned[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
