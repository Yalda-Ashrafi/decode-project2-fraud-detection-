# 05_evaluate.py
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# --- reload data and split identically (same random_state = same split) ---
df = pd.read_csv("creditcard.csv")
df["Hour"] = (df["Time"] / 3600) % 24
df = df.drop(columns=["Time"])
X, y = df.drop(columns=["Class"]), df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

lr_pipeline = joblib.load("models/lr_pipeline.pkl")
rf_pipeline = joblib.load("models/rf_pipeline.pkl")


def evaluate(model, name, X_test, y_test):
    """Print the metrics that actually matter for imbalanced data."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]   # probability of class 1

    metrics = {
        "Precision":  precision_score(y_test, y_pred, zero_division=0),
        "Recall":     recall_score(y_test, y_pred),
        "F1":         f1_score(y_test, y_pred),
        "ROC-AUC":    roc_auc_score(y_test, y_proba),
        "PR-AUC":     average_precision_score(y_test, y_proba),
    }

    print(f"\n{'='*45}\n{name}\n{'='*45}")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"\nTrue Negatives : {tn:,}   False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}       True Positives : {tp:,}")
    print(f"\nFraud caught: {tp}/{tp+fn}   Fraud missed: {fn}")
    print("\n", classification_report(y_test, y_pred,
          target_names=["Legitimate", "Fraud"], digits=4))

    return metrics, y_proba, y_pred


lr_metrics, lr_proba, lr_pred = evaluate(lr_pipeline, "LOGISTIC REGRESSION", X_test, y_test)
rf_metrics, rf_proba, rf_pred = evaluate(rf_pipeline, "RANDOM FOREST",       X_test, y_test)

def plot_confusion(y_true, y_pred, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred: Legit", "Pred: Fraud"])
    ax.set_yticks([0, 1], ["True: Legit", "True: Fraud"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13, fontweight="bold")
    return im

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
plot_confusion(y_test, lr_pred, "Logistic Regression", axes[0])
plot_confusion(y_test, rf_pred, "Random Forest",       axes[1])
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()

plt.figure(figsize=(7, 6))
for proba, name, colour in [(lr_proba, "Logistic Regression", "#e76f51"),
                            (rf_proba, "Random Forest",       "#2a9d8f")]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, color=colour, lw=2, label=f"{name} (AUC = {auc:.4f})")

plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve — Fraud Detection")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("roc_curve.png", dpi=150)
plt.show()

plt.figure(figsize=(7, 6))
for proba, name, colour in [(lr_proba, "Logistic Regression", "#e76f51"),
                            (rf_proba, "Random Forest",       "#2a9d8f")]:
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    plt.plot(recall, precision, color=colour, lw=2, label=f"{name} (AP = {ap:.4f})")

baseline = y_test.mean()   # 0.0017
plt.axhline(baseline, ls="--", color="grey", lw=1,
            label=f"No-skill baseline ({baseline:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve — Fraud Detection")
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
plt.savefig("pr_curve.png", dpi=150)
print("Saved plot to:", os.path.abspath("confusion_matrices.png"))
print("Saved plot to:", os.path.abspath("roc_curve.png"))
print("Saved plot to:", os.path.abspath("pr_curve.png"))

plt.show()

precision, recall, thresholds = precision_recall_curve(y_test, rf_proba)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_idx = np.argmax(f1_scores[:-1])
best_threshold = thresholds[best_idx]

print(f"Best threshold: {best_threshold:.4f}")
print(f"  → Precision {precision[best_idx]:.4f}, Recall {recall[best_idx]:.4f}, "
      f"F1 {f1_scores[best_idx]:.4f}")

y_pred_tuned = (rf_proba >= best_threshold).astype(int)
print(confusion_matrix(y_test, y_pred_tuned))