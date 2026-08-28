"""
Task 2 — Logistic Regression: Diabetes Prediction
------------------------------------------------------------
Predicts whether a patient has diabetes (1) or not (0) using the
Pima Indians Diabetes Dataset.

Steps: load data -> assign column names -> check missing/zero values
-> 80/20 split -> feature scaling -> train Logistic Regression
-> evaluate -> interpret coefficients.

Run:  python src/02_diabetes_logistic_regression.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# 1. Load dataset --------------------------------------------------------
# Public mirror of the Pima Indians Diabetes Dataset (UCI).
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

# 2. Check for missing / zero values -------------------------------------
# In this dataset, 0 is not physiologically valid for these columns,
# so a zero really means "missing".
zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("\nZero-value counts (treated as missing):")
print((df[zero_invalid_cols] == 0).sum())

df[zero_invalid_cols] = df[zero_invalid_cols].replace(0, np.nan)
df[zero_invalid_cols] = df[zero_invalid_cols].fillna(df[zero_invalid_cols].median())
print("\nMissing values after imputation:", df.isnull().sum().sum())

# 3. Train / test split ---------------------------------------------------
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature scaling --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Logistic Regression -----------------------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Predict & evaluate -----------------------------------------------
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n=== Logistic Regression — Diabetes Prediction ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Interpret coefficients ---------------------------------------------
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False)
print("\nModel coefficients (standardized features, sorted by impact):")
print(coef_df.to_string(index=False))

# 8. Save metrics -------------------------------------------------------
metrics = {
    "task": "Logistic Regression - Diabetes",
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1_score": round(f1, 4),
    "roc_auc": round(roc_auc, 4),
}
with open(f"{OUT}/02_diabetes_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# 9. Plots ----------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.title("Confusion Matrix — Logistic Regression (Diabetes)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT}/02_confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="#EC4899", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Logistic Regression (Diabetes)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/02_roc_curve.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4.5))
sns.barplot(x="coefficient", y="feature", data=coef_df, color="#EC4899")
plt.title("Logistic Regression Coefficients (standardized)")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(f"{OUT}/02_coefficients.png", dpi=150)
plt.close()

print(f"\nSaved plots and metrics to '{OUT}/'.")
