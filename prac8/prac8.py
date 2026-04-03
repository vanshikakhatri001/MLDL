import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
def load_data(file_path):
    print(f"Loading dataset from {file_path}...")
    data = pd.read_csv(file_path)

    # Extract labels and pixel data
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values

    # Normalize pixel values (0-255 to 0.0-1.0)
    X = X.astype('float32') / 255.0

    # Reshape for CNN: (Batch, Height, Width, Channels)
    X = X.reshape(-1, 28, 28, 1)

    return X, y

# Load your uploaded file
try:
    # If only test file exists, we split it
    X_test, y_test = load_data('mnist_test.csv')

    # In full experiment, replace with:
    # X_train, y_train = load_data('mnist_train.csv')

    X_train, X_val, y_train, y_val = train_test_split(
        X_test, y_test, test_size=0.2, random_state=42
    )

    print("Data prepared successfully.")

except FileNotFoundError:
    print("Error: mnist_test.csv not found. Please ensure the file is in the working directory.")

# ==========================================
# 2. BASELINE CNN DESIGN (No Tuning)
# ==========================================
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten and Fully Connected Layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes (0–9)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 3. TRAINING
# ==========================================
print("\nStarting Training...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    batch_size=32,
    verbose=1
)

# ==========================================
# 4. PERFORMANCE ANALYTICAL METRICS
# ==========================================
print("\n" + "="*30)
print("PERFORMANCE ANALYTICAL METRICS")
print("="*30)

# Evaluate on validation set
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)

print(f"Final Validation Loss: {loss:.4f}")
print(f"Final Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Detailed Classification Report
y_pred = np.argmax(model.predict(X_val), axis=1)

print("\nDetailed Metrics:")
print(classification_report(y_val, y_pred))

# ==========================================
# 5. VISUAL OUTPUTS
# ==========================================

# Visual 1: Training History Graph
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy Performance')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss Performance')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Visual 2: Actual Prediction Results (Image Grid)
plt.figure(figsize=(12, 8))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    img = X_val[i].reshape(28, 28)
    actual_label = y_val[i]
    pred_label = y_pred[i]

    color = 'green' if actual_label == pred_label else 'red'

    plt.imshow(img, cmap='gray')
    plt.title(f"Act: {actual_label} | Pred: {pred_label}", color=color)
    plt.axis('off')

plt.suptitle("Sample Prediction Results from CSV Data")
plt.tight_layout()
plt.show()
