import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------
# 1. LOAD DATA
# ----------------------------------

df = pd.read_csv("data/housing.csv")

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# One-hot encode categorical feature
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

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
# 4. TRAIN SVM REGRESSOR
# ----------------------------------

model = SVR(
    kernel="rbf",
    C=100,
    epsilon=0.1
)

model.fit(X_train_scaled, y_train)

# ----------------------------------
# 5. PREDICTION
# ----------------------------------

y_pred = model.predict(X_test_scaled)

# ----------------------------------
# 6. EVALUATION
# ----------------------------------

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# ----------------------------------
# 7. ACTUAL vs PREDICTED
# ----------------------------------

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("SVR: Actual vs Predicted")
plt.grid(True)
plt.show()
