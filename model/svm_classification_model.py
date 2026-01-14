import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------------
# 1. LOAD DATA
# ----------------------------------

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target   # 0 = malignant, 1 = benign

# ----------------------------------
# 2. TRAIN-TEST SPLIT
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# 3. FEATURE SCALING (MANDATORY)
# ----------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------
# 4. TRAIN SVM CLASSIFIER
# ----------------------------------

model = SVC(
    kernel="rbf",   # try: linear, rbf, poly
    C=1.0,
    gamma="scale"
)

model.fit(X_train_scaled, y_train)

# ----------------------------------
# 5. PREDICTION
# ----------------------------------

y_pred = model.predict(X_test_scaled)

# ----------------------------------
# 6. EVALUATION
# ----------------------------------

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------------
# 7. SUPPORT VECTORS
# ----------------------------------

print("Number of support vectors:", model.n_support_)
