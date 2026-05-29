# ======================
# Import Libraries
# ======================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier

# ======================
# Load Dataset
# ======================

# Download dataset from:
# https://archive.ics.uci.edu/ml/datasets/bank+marketing

# Replace with your CSV file name
df = pd.read_csv("bank.csv", sep=';')

# ======================
# Display Dataset
# ======================

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names:\n")
print(df.columns)

# ======================
# Check Missing Values
# ======================

print("\nMissing Values:\n")
print(df.isnull().sum())

# ======================
# Encode Categorical Columns
# ======================

le = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# ======================
# Features and Target
# ======================

X = df.drop("y", axis=1)
y = df["y"]

# ======================
# Train Test Split
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================
# Feature Scaling
# ======================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# Train Model
# ======================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ======================
# Predictions
# ======================

y_pred = model.predict(X_test)

# ======================
# Evaluation
# ======================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ======================
# Confusion Matrix
# ======================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ======================
# Feature Importance
# ======================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:\n")
print(importance)

# ======================
# Plot Feature Importance
# ======================

plt.figure(figsize=(10,6))

importance.plot(kind='bar')

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.show()

# ======================
# Sample Prediction
# ======================

sample = X.iloc[0:1]

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("\nSample Prediction:", prediction)

if prediction[0] == 1:
    print("Customer will subscribe to term deposit")
else:
    print("Customer will NOT subscribe to term deposit")