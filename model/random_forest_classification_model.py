import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------------
# 1. LOAD DATA
# ----------------------------------
df = pd.read_csv("data/breast_cancer.csv")

# Convert target column
df["diagnosis"] = df["diagnosis"].map({"M": 0, "B": 1})

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ----------------------------------
# 2. TRAIN-TEST SPLIT
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# 3. TRAIN RANDOM FOREST CLASSIFIER
# ----------------------------------
model = RandomForestClassifier(
    n_estimators=100,     # number of trees
    max_depth=5,          # controls overfitting
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------
# 4. PREDICT & EVALUATE
# ----------------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
