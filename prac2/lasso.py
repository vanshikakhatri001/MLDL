# ======================================
# STEP 1: IMPORT LIBRARIES
# ======================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ======================================
# STEP 2: LOAD DATASET
# ======================================
data = pd.read_csv("USA_Housing.csv")

# ======================================
# STEP 3: FEATURE SELECTION
# ======================================
X = data.drop(['Price', 'Address'], axis=1)
y = data['Price']

# ======================================
# STEP 4: TRAIN-TEST SPLIT
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
# LASSO REGRESSION (BASE MODEL)
# ======================================
lasso_model = Lasso(max_iter=5000)
lasso_model.fit(X_train_scaled, y_train)

pred_default = lasso_model.predict(X_test_scaled)

rmse_default = np.sqrt(mean_squared_error(y_test, pred_default))
r2_default = r2_score(y_test, pred_default)

print("===== LASSO REGRESSION (DEFAULT) =====")
print(f"RMSE: {rmse_default:.2f} USD")
print(f"R2 Score: {r2_default:.4f}")

# ======================================
# LASSO REGRESSION (WITH TUNING)
# ======================================
param_grid = {
    'alpha': np.logspace(-4, 1, 50)
}

grid_search = GridSearchCV(
    Lasso(max_iter=5000),
    param_grid,
    cv=5,
    scoring='r2'
)

grid_search.fit(X_train_scaled, y_train)

best_lasso_model = grid_search.best_estimator_

pred_tuned = best_lasso_model.predict(X_test_scaled)

rmse_tuned = np.sqrt(mean_squared_error(y_test, pred_tuned))
r2_tuned = r2_score(y_test, pred_tuned)

print("\n===== LASSO REGRESSION (TUNED) =====")
print("Best Alpha:", grid_search.best_params_)
print(f"RMSE: {rmse_tuned:.2f} USD")
print(f"R2 Score: {r2_tuned:.4f}")

# ======================================
# FEATURE IMPORTANCE (LASSO)
# ======================================
feature_importance = pd.Series(
    best_lasso_model.coef_,
    index=X.columns
).abs().sort_values(ascending=False)

print("\nTop 5 Important Features (Lasso):")
print(feature_importance.head(5))

# ======================================
# VISUALIZATION
# ======================================
plt.figure(figsize=(10, 5))

feature_importance.head(10).plot(kind='bar')

plt.title("Top 10 Features based on Lasso Coefficients")
plt.ylabel("Absolute Coefficient Magnitude")

plt.show()
