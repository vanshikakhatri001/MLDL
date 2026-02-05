import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# =====================================================
# LINEAR REGRESSION – Advertising Sales Dataset
# =====================================================

print("\n--- LINEAR REGRESSION: SALES PREDICTION ---")

sales_data = pd.read_csv(
    r"C:\Users\vrkha\Desktop\MLDL\advertising.csv"
)
print(sales_data.head())
X_lr = sales_data[['TV']]
y_lr = sales_data['Sales']

# Graph 1: Actual data distribution
plt.scatter(X_lr, y_lr)
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.title("Actual Data: TV Budget vs Sales")
plt.show()

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_lr, y_lr, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)

y_pred_lr = lr_model.predict(X_test_lr)

print("MSE:", mean_squared_error(y_test_lr, y_pred_lr))
print("R2 Score:", r2_score(y_test_lr, y_pred_lr))

# Graph 2: Regression line
plt.scatter(X_test_lr, y_test_lr)
plt.plot(X_test_lr, y_pred_lr)
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.title("Linear Regression Model Output")
plt.show()


# =====================================================
# LOGISTIC REGRESSION – Breast Cancer Dataset
# =====================================================

print("\n--- LOGISTIC REGRESSION: BREAST CANCER PREDICTION ---")

data = pd.read_csv(
    r"C:\Users\vrkha\Desktop\MLDL\data.csv"
)

data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

X_log = data.drop('diagnosis', axis=1)
y_log = data['diagnosis']

# Graph 3: Class distribution
plt.hist(y_log)
plt.xlabel("Class (0 = Benign, 1 = Malignant)")
plt.ylabel("Count")
plt.title("Class Distribution")
plt.show()

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42
)

log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train_log, y_train_log)

y_pred_log = log_model.predict(X_test_log)
y_prob_log = log_model.predict_proba(X_test_log)[:, 1]

print("Accuracy:", accuracy_score(y_test_log, y_pred_log))

# Graph 4: Confusion matrix (visual)
cm = confusion_matrix(y_test_log, y_pred_log)
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Graph 5: ROC curve
fpr, tpr, _ = roc_curve(y_test_log, y_prob_log)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="AUC = %.2f" % roc_auc)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
