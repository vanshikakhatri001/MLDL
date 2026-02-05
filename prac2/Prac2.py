import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv(
    r"C:\Users\vrkha\Desktop\MLDL\vehicle_co2_emissions.csv"
)

print(df.columns)   # <-- verify columns once

# ===============================
# FEATURES & TARGET
# ===============================
X = df[[
    'Engine Size(L)',
    'Cylinders',
    'Fuel Consumption City (L/100 km)',
    'Fuel Consumption Hwy (L/100 km)',
    'Fuel Consumption Comb (L/100 km)'
]]

y = df['CO2 Emissions(g/km)']

# ===============================
# TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# MULTIPLE LINEAR REGRESSION
# ===============================
mlr = LinearRegression()
mlr.fit(X_train, y_train)
y_pred_mlr = mlr.predict(X_test)

print("\nMultiple Linear Regression")
print("MSE:", mean_squared_error(y_test, y_pred_mlr))
print("R2:", r2_score(y_test, y_pred_mlr))

# ===============================
# RIDGE REGRESSION
# ===============================
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

print("\nRidge Regression")
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("R2:", r2_score(y_test, y_pred_ridge))

# ===============================
# LASSO REGRESSION
# ===============================
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

print("\nLasso Regression")
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("R2:", r2_score(y_test, y_pred_lasso))


# ===============================
# GRAPH 1: MULTIPLE LINEAR REGRESSION
# ===============================
plt.scatter(y_test, y_pred_mlr)
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Multiple Linear Regression")
plt.show()

# ===============================
# GRAPH 2: RIDGE REGRESSION
# ===============================
plt.scatter(y_test, y_pred_ridge)
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Ridge Regression")
plt.show()

# ===============================
# GRAPH 3: LASSO REGRESSION
# ===============================
plt.scatter(y_test, y_pred_lasso)
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Lasso Regression")
plt.show()

# ===============================
# COMPARISON GRAPH (ONE FIGURE)
# ===============================
plt.scatter(y_test, y_pred_mlr, label="Multiple Regression")
plt.scatter(y_test, y_pred_ridge, label="Ridge Regression")
plt.scatter(y_test, y_pred_lasso, label="Lasso Regression")

plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Model Comparison: CO2 Emission Prediction")
plt.legend()
plt.show()
