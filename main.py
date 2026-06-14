# -------------------------------------
# IMPORT LIBRARIES
# -------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_score,
    recall_score, f1_score
)

# -------------------------------------
# STEP 1: LOAD DATASET
# -------------------------------------
data = pd.read_csv("dataset/pump_data.csv")
print("Dataset Loaded Successfully ")

# -------------------------------------
# STEP 2: REMOVE UNNECESSARY COLUMNS
# -------------------------------------
data = data.drop(["UDI", "Product ID"], axis=1)
data = data.drop(["TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)

# -------------------------------------
# STEP 3: ENCODING
# -------------------------------------
data = pd.get_dummies(data, columns=["Type"], drop_first=True)

# -------------------------------------
# STEP 3.1: CORRELATION HEATMAP (A.2)
# -------------------------------------
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------------------------
# STEP 3.2: DISTRIBUTION PLOTS (A.3)
# -------------------------------------
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

data[features].hist(figsize=(10,8))
plt.suptitle("Feature Distributions")
plt.show()

# -------------------------------------
# STEP 4: DEFINE X AND y
# -------------------------------------
X = data.drop("Machine failure", axis=1)
y = data["Machine failure"]

# -------------------------------------
# STEP 5: TRAIN TEST SPLIT
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------
# MODEL 1: LOGISTIC REGRESSION
# -------------------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# -------------------------------------
# MODEL 2: DECISION TREE
# -------------------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# -------------------------------------
# MODEL 3: RANDOM FOREST
# -------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# -------------------------------------
# STEP 6: METRICS
# -------------------------------------
log_acc = accuracy_score(y_test, y_pred_log)
dt_acc = accuracy_score(y_test, y_pred_dt)
rf_acc = accuracy_score(y_test, y_pred_rf)

print("\n========== Accuracy ==========")
print("Logistic:", log_acc)
print("Decision Tree:", dt_acc)
print("Random Forest:", rf_acc)

# -------------------------------------
# STEP 7: CONFUSION MATRIX (A.4)
# -------------------------------------
cm_log = confusion_matrix(y_test, y_pred_log)
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_rf = confusion_matrix(y_test, y_pred_rf)

fig, axes = plt.subplots(1, 3, figsize=(15,4))

sns.heatmap(cm_log, annot=True, fmt='d', ax=axes[0])
axes[0].set_title("Logistic Regression")

sns.heatmap(cm_dt, annot=True, fmt='d', ax=axes[1])
axes[1].set_title("Decision Tree")

sns.heatmap(cm_rf, annot=True, fmt='d', ax=axes[2])
axes[2].set_title("Random Forest")

plt.tight_layout()
plt.show()

# -------------------------------------
# STEP 8: PERFORMANCE GRAPH (A.5)
# -------------------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

log_metrics = [
    accuracy_score(y_test, y_pred_log),
    precision_score(y_test, y_pred_log),
    recall_score(y_test, y_pred_log),
    f1_score(y_test, y_pred_log)
]

dt_metrics = [
    accuracy_score(y_test, y_pred_dt),
    precision_score(y_test, y_pred_dt),
    recall_score(y_test, y_pred_dt),
    f1_score(y_test, y_pred_dt)
]

rf_metrics = [
    accuracy_score(y_test, y_pred_rf),
    precision_score(y_test, y_pred_rf),
    recall_score(y_test, y_pred_rf),
    f1_score(y_test, y_pred_rf)
]

x = np.arange(len(metrics))

plt.figure(figsize=(8,5))
plt.bar(x - 0.2, log_metrics, width=0.2, label="Logistic")
plt.bar(x, dt_metrics, width=0.2, label="Decision Tree")
plt.bar(x + 0.2, rf_metrics, width=0.2, label="Random Forest")

plt.xticks(x, metrics)
plt.title("Model Performance Comparison")
plt.legend()
plt.show()

# -------------------------------------
# STEP 9: FEATURE IMPORTANCE (A.6)
# -------------------------------------
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

feature_importances.sort_values().plot(kind='barh')
plt.title("Feature Importance - Random Forest")
plt.show()

# -------------------------------------
# STEP 10: REPORT
# -------------------------------------
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# -------------------------------------
# STEP 11: SAVE MODEL
# -------------------------------------
with open("model/random_forest_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

print("\nModel saved successfully ")