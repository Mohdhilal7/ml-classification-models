"""
Task 1 — Random Forest Classifier: Breast Cancer Diagnosis
------------------------------------------------------------
Predicts whether a tumor is malignant or benign using the
Breast Cancer Wisconsin dataset (built into scikit-learn).

Evaluates with: Accuracy, Confusion Matrix, Precision, Recall,
F1-score, and ROC-AUC.

Run:  python src/01_breast_cancer_random_forest.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

OUT = "outputs"
import os
os.makedirs(OUT, exist_ok=True)

# 1. Load data ---------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape}")
print(f"Class balance:\n{y.value_counts()}\n")

# 2. Train / test split -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train Random Forest -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# 4. Predict & evaluate ---------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("=== Random Forest — Breast Cancer Classification ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# 5. Save metrics as JSON (used by README generator) ----------------------
metrics = {
    "task": "Random Forest - Breast Cancer",
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1_score": round(f1, 4),
    "roc_auc": round(roc_auc, 4),
}
with open(f"{OUT}/01_breast_cancer_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 6. Plots ------------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Confusion Matrix — Random Forest (Breast Cancer)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT}/01_confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="#A855F7", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Random Forest (Breast Cancer)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/01_roc_curve.png", dpi=150)
plt.close()

# Feature importance (top 10)
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(6, 4.5))
sns.barplot(x=importances.values, y=importances.index, color="#A855F7")
plt.title("Top 10 Feature Importances — Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/01_feature_importance.png", dpi=150)
plt.close()

print(f"\nSaved plots and metrics to '{OUT}/'.")
