
import numpy as np
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)
from src.config import THRESHOLD

def evaluate_classifier(model, X_val, y_val, threshold=THRESHOLD):
    # probability for ROC-AUC
    proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)

    # threshold-based prediction
    pred = (proba >= threshold).astype(int)

    cm = confusion_matrix(y_val, pred)
    report = classification_report(y_val, pred, digits=4)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, pred, average="binary", zero_division=0
    )

    results = {
        "roc_auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "report": report,
    }
    return results
