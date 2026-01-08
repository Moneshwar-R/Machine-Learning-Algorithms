import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
import matplotlib.pyplot as plt

# ----------------------------------
# 1. LOAD DATA
# ----------------------------------
df = pd.read_csv("data/housing.csv")

# ----------------------------------
# 2. HANDLE MISSING VALUES
# ----------------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)

# ----------------------------------
# 3. ONE-HOT ENCODE CATEGORICAL DATA
# ----------------------------------
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# ----------------------------------
# 4. SPLIT FEATURES & TARGET
# ----------------------------------
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# ----------------------------------
# 5. TRAIN-TEST SPLIT
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# 6. TRAIN DECISION TREE REGRESSOR
# ----------------------------------
model = DecisionTreeRegressor(
    max_depth=6,        # controls overfitting
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------
# 7. PREDICTION
# ----------------------------------
y_pred = model.predict(X_test)

# ----------------------------------
# 8. EVALUATION
# ----------------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ----------------------------------
# 9. VISUALIZE TREE (OPTIONAL)
# ----------------------------------
plt.figure(figsize=(20, 8))
tree.plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    max_depth=3   # limit visualization depth
)
plt.show()
