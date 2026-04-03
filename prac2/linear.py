# ==============================
# STEP 1: IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ==============================
# STEP 2: LOAD DATASET
# ==============================
data = pd.read_csv("USA_Housing.csv")

# ==============================
# STEP 3: DEFINE FEATURES & TARGET
# ==============================
features = data.drop(['Price', 'Address'], axis=1)
target = data['Price']

# ==============================
# STEP 4: SPLIT DATA
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# ==============================
# STEP 5: FEATURE SCALING
# ==============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# STEP 6: TRAIN MODEL
# ==============================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ==============================
# STEP 7: MAKE PREDICTIONS
# ==============================
predictions = model.predict(X_test_scaled)

# ==============================
# STEP 8: EVALUATE MODEL
# ==============================
rmse_value = np.sqrt(mean_squared_error(y_test, predictions))
r2_value = r2_score(y_test, predictions)

print("===== MULTIPLE LINEAR REGRESSION OUTPUT =====")
print("RMSE:", rmse_value)
print("R2 Score:", r2_value)

# ==============================
# STEP 9: VISUALIZATION
# ==============================
plt.figure(figsize=(8, 6))

sns.scatterplot(x=y_test, y=predictions)

# Ideal line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted Prices (Multiple Linear Regression)")

plt.show()
