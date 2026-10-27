"""
Breast Cancer Detection with Machine Learning
Dataset: Breast Cancer Wisconsin (Diagnostic) — 569 samples, 30 features
Models : Logistic Regression, Random Forest, SVM, XGBoost
Author : Muhammad Awais
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

RANDOM_STATE = 42

# 1. Load data  (target: 0 = malignant, 1 = benign -> flip so 1 = malignant)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(1 - data.target, name="malignant")
print(f"Samples: {len(X)} | Features: {X.shape[1]} | Malignant: {y.sum()} | Benign: {(y == 0).sum()}")

# 2. Train / test split (stratified 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# 3. Models
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE)),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                             eval_metric="logloss", random_state=RANDOM_STATE),
}

# 4. Train, cross-validate and evaluate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = []
for name, model in models.items():
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    results.append({
        "model": name,
        "cv_accuracy": round(cv_acc.mean() * 100, 2),
        "accuracy": round(accuracy_score(y_test, pred) * 100, 2),
        "precision": round(precision_score(y_test, pred) * 100, 2),
        "recall": round(recall_score(y_test, pred) * 100, 2),
        "f1": round(f1_score(y_test, pred) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    })

df = pd.DataFrame(results).sort_values(["f1", "roc_auc"], ascending=False)
print("\n", df.to_string(index=False))

# 5. Top features (Random Forest importance)
rf = models["Random Forest"]
top = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
print("\nTop 5 features:\n", top.round(4).to_string())

# 6. Save outputs
df.to_csv("results.csv", index=False)
with open("results.json", "w") as f:
    json.dump({"results": results, "top_features": top.round(4).to_dict(),
               "test_size": len(y_test)}, f, indent=2)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(df["model"][::-1], df["f1"][::-1], color="#2DD4BF")
ax.set_xlim(85, 100)
ax.set_xlabel("F1-score (%) on test set")
ax.set_title("Breast Cancer Detection — model comparison")
for i, v in enumerate(df["f1"][::-1]):
    ax.text(v + 0.2, i, f"{v:.2f}%", va="center")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
print("\nSaved: results.csv, results.json, model_comparison.png")
