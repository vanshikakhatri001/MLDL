# ======================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ======================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ======================================
# STEP 2: LOAD DATASET
# ======================================
data = pd.read_csv("USA_Housing.csv")

# ======================================
# STEP 3: PREPARE FEATURES & TARGET
# ======================================
X = data.drop(['Price', 'Address'], axis=1)
y = data['Price']

# ======================================
# STEP 4: SPLIT DATA INTO TRAIN & TEST
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================
# STEP 5: FEATURE SCALING
# ======================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================================
# RIDGE REGRESSION (BASE MODEL)
# ======================================
ridge_model = Ridge()
ridge_model.fit(X_train_scaled, y_train)

pred_default = ridge_model.predict(X_test_scaled)

rmse_default = np.sqrt(mean_squared_error(y_test, pred_default))
r2_default = r2_score(y_test, pred_default)

print("===== RIDGE REGRESSION (DEFAULT) =====")
print(f"RMSE: {rmse_default:.2f} USD")
print(f"R2 Score: {r2_default:.4f}")

# ======================================
# RIDGE REGRESSION (WITH TUNING)
# ======================================
param_grid = {
    'alpha': np.logspace(-2, 2, 50)
}

grid_search = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring='r2'
)

grid_search.fit(X_train_scaled, y_train)

best_ridge_model = grid_search.best_estimator_

pred_tuned = best_ridge_model.predict(X_test_scaled)

rmse_tuned = np.sqrt(mean_squared_error(y_test, pred_tuned))
r2_tuned = r2_score(y_test, pred_tuned)

print("\n===== RIDGE REGRESSION (TUNED) =====")
print("Best Alpha:", grid_search.best_params_)
print(f"RMSE: {rmse_tuned:.2f} USD")
print(f"R2 Score: {r2_tuned:.4f}")

# ======================================
# FEATURE IMPORTANCE (RIDGE)
# ======================================
feature_importance = pd.Series(
    best_ridge_model.coef_,
    index=X.columns
).abs().sort_values(ascending=False)

print("\nTop 5 Important Features (Ridge):")
print(feature_importance.head(5))

# ======================================
# VISUALIZATION OF COEFFICIENTS
# ======================================
plt.figure(figsize=(10, 5))

feature_importance.head(10).plot(kind='bar')

plt.title("Top 10 Features based on Ridge Coefficients")
plt.ylabel("Absolute Coefficient Magnitude")

plt.show()
