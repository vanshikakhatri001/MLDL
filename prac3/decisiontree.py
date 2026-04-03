# Decision Tree Classifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load Dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Data Preprocessing
df.drop("id", axis=1, inplace=True)

# Handle missing values (BMI)
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

# Encode categorical variables
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# Features & Target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Decision Tree Model
dtree = DecisionTreeClassifier(random_state=42)

# Reduced Hyperparameter Grid (LESS LOAD)
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 5]
}

# Grid Search (SAFE SETTINGS)
grid_search = GridSearchCV(
    estimator=dtree,
    param_grid=param_grid,
    cv=3,
    n_jobs=1,
    verbose=1
)

# Train Model
grid_search.fit(X_train, y_train)

print("===== BEST PARAMETERS =====")
print(grid_search.best_params_)
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# Best Model
best_model = grid_search.best_estimator_

# Prediction
y_pred = best_model.predict(X_test)

# Evaluation
print("\n===== FINAL MODEL EVALUATION =====")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Stroke", "Stroke"]
)

fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(cmap="Blues", ax=ax)
plt.title("Confusion Matrix: Stroke Prediction")
plt.show()

# Tree Visualization (Top 3 Levels)
plt.figure(figsize=(22, 10))
plot_tree(
    best_model,
    max_depth=3,
    feature_names=X.columns,
    class_names=["No Stroke", "Stroke"],
    filled=True,
    rounded=True,
    fontsize=11
)
plt.title("Decision Tree Structure (Top 3 Levels)")
plt.show()

# Feature Importance
importances = best_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance_df,
    palette="viridis",
    hue="Feature",
    legend=False
)

plt.title("Feature Importance for Stroke Prediction")
plt.show()
