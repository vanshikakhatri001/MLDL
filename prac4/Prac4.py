import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv(
    r"C:\Users\vrkha\Desktop\MLDL\breast_cancer.csv"
)

print(df.head())
print(df.shape)

# ===============================
# DATA CLEANING
# ===============================
# Drop unnecessary columns
df = df.drop(['id', 'Unnamed: 32'], axis=1)

# Encode target variable
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# ===============================
# FEATURES & TARGET
# ===============================
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# ===============================
# FEATURE SCALING (VERY IMPORTANT FOR KNN)
# ===============================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ===============================
# TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# KNN CLASSIFIER
# ===============================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# ===============================
# PERFORMANCE EVALUATION
# ===============================
print("\n--- KNN Classifier ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ===============================
# ACCURACY vs K GRAPH
# ===============================
k_values = range(1, 21)
accuracy_list = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred_k = model.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, pred_k))

plt.plot(k_values, accuracy_list, marker='o')
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy for Different K Values")
plt.show()
