"""
Task 3 — XGBoost Classifier: Titanic Survival Prediction
------------------------------------------------------------
Predicts whether a passenger survived the Titanic disaster.

Steps: load data -> assign column names -> check missing values
-> 80/20 split -> feature scaling -> train XGBoost (with params)
-> evaluate -> interpret feature importance.

Requires: pip install xgboost
Run:  python src/03_titanic_xgboost.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from xgboost import XGBClassifier

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# 1. Load dataset ----------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(DATA_URL)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# 2. Check for missing values ----------------------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

# 3. Clean & engineer features -----------------------------------------
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize"]
X = df[features]
y = df["Survived"]

# 4. Train / test split -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Feature scaling ---------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train XGBoost with explicit hyperparameters --------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train_scaled, y_train)

# 7. Predict & evaluate -----------------------------------------------
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n=== XGBoost — Titanic Survival Prediction ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Feature importance --------------------------------------------------
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importance:\n", importances)

# 9. Save metrics -------------------------------------------------------
metrics = {
    "task": "XGBoost - Titanic",
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1_score": round(f1, 4),
    "roc_auc": round(roc_auc, 4),
}
with open(f"{OUT}/03_titanic_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 10. Plots -----------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Did not survive", "Survived"], yticklabels=["Did not survive", "Survived"])
plt.title("Confusion Matrix — XGBoost (Titanic)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT}/03_confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="#F43F5E", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — XGBoost (Titanic)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/03_roc_curve.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4.5))
sns.barplot(x=importances.values, y=importances.index, color="#F43F5E")
plt.title("Feature Importance — XGBoost (Titanic)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/03_feature_importance.png", dpi=150)
plt.close()

print(f"\nSaved plots and metrics to '{OUT}/'.")
