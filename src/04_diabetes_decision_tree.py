"""
Task 4 — Decision Tree Classifier: Diabetes Prediction
------------------------------------------------------------
Predicts whether a patient has diabetes (0 = No, 1 = Yes) using the
Pima Indians Diabetes Dataset.

Steps: load data -> assign column names -> handle missing/unrealistic
zero values -> 80/20 split (random_state=42) -> define X, y ->
train DecisionTreeClassifier -> evaluate -> train a depth-restricted
tree (max_depth=3) -> compare both -> feature importance.

Run:  python src/04_diabetes_decision_tree.py
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# 1. Load dataset & assign column names ----------------------------------
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
LOCAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pima-diabetes.csv")
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
if os.path.exists(LOCAL_PATH):
    df = pd.read_csv(LOCAL_PATH, names=COLUMNS)
else:
    df = pd.read_csv(DATA_URL, names=COLUMNS)
print(f"Dataset shape: {df.shape}")

# 2. Handle missing / unrealistic zero values -----------------------------
zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("\nUnrealistic zero-value counts:")
print((df[zero_invalid_cols] == 0).sum())
df[zero_invalid_cols] = df[zero_invalid_cols].replace(0, np.nan)
df[zero_invalid_cols] = df[zero_invalid_cols].fillna(df[zero_invalid_cols].median())

# 3. Train / test split (80/20, random_state=42) --------------------------
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train a full Decision Tree -----------------------------------------
tree_full = DecisionTreeClassifier(random_state=42)
tree_full.fit(X_train, y_train)
pred_full = tree_full.predict(X_test)

acc_full = accuracy_score(y_test, pred_full)
prec_full = precision_score(y_test, pred_full)
rec_full = recall_score(y_test, pred_full)
f1_full = f1_score(y_test, pred_full)
cm_full = confusion_matrix(y_test, pred_full)

print("\n=== Decision Tree (full depth) ===")
print(f"Accuracy : {acc_full:.4f}")
print(f"Precision: {prec_full:.4f}")
print(f"Recall   : {rec_full:.4f}")
print(f"F1-score : {f1_full:.4f}")
print("Confusion Matrix:\n", cm_full)

# 5. Train a depth-restricted Decision Tree (max_depth=3) ----------------
tree_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_shallow.fit(X_train, y_train)
pred_shallow = tree_shallow.predict(X_test)

acc_shallow = accuracy_score(y_test, pred_shallow)
prec_shallow = precision_score(y_test, pred_shallow)
rec_shallow = recall_score(y_test, pred_shallow)
f1_shallow = f1_score(y_test, pred_shallow)
cm_shallow = confusion_matrix(y_test, pred_shallow)

print("\n=== Decision Tree (max_depth=3) ===")
print(f"Accuracy : {acc_shallow:.4f}")
print(f"Precision: {prec_shallow:.4f}")
print(f"Recall   : {rec_shallow:.4f}")
print(f"F1-score : {f1_shallow:.4f}")
print("Confusion Matrix:\n", cm_shallow)

# 6. Compare both models --------------------------------------------------
comparison = pd.DataFrame({
    "Model": ["Full Tree", "Max Depth = 3"],
    "Accuracy": [acc_full, acc_shallow],
    "Precision": [prec_full, prec_shallow],
    "Recall": [rec_full, rec_shallow],
    "F1-score": [f1_full, f1_shallow],
})
print("\nModel comparison:\n", comparison.to_string(index=False))

# 7. Feature importance ----------------------------------------------
importances = pd.Series(tree_full.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importance (full tree):\n", importances)

# 8. Save metrics -------------------------------------------------------
metrics = {
    "task": "Decision Tree - Diabetes",
    "full_tree": {"accuracy": round(acc_full, 4), "precision": round(prec_full, 4),
                  "recall": round(rec_full, 4), "f1_score": round(f1_full, 4)},
    "max_depth_3": {"accuracy": round(acc_shallow, 4), "precision": round(prec_shallow, 4),
                     "recall": round(rec_shallow, 4), "f1_score": round(f1_shallow, 4)},
}
with open(f"{OUT}/04_decision_tree_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 9. Plots ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(cm_full, annot=True, fmt="d", cmap="Purples", ax=axes[0],
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
axes[0].set_title("Full Tree")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")
sns.heatmap(cm_shallow, annot=True, fmt="d", cmap="Oranges", ax=axes[1],
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
axes[1].set_title("Max Depth = 3")
axes[1].set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT}/04_confusion_matrices.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4.5))
sns.barplot(x=importances.values, y=importances.index, color="#6366F1")
plt.title("Feature Importance — Decision Tree (Diabetes)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/04_feature_importance.png", dpi=150)
plt.close()

plt.figure(figsize=(14, 7))
plot_tree(tree_shallow, feature_names=X.columns, class_names=["No Diabetes", "Diabetes"],
          filled=True, rounded=True, fontsize=9)
plt.title("Decision Tree (max_depth=3)")
plt.tight_layout()
plt.savefig(f"{OUT}/04_tree_visualization.png", dpi=150)
plt.close()

print(f"\nSaved plots and metrics to '{OUT}/'.")
