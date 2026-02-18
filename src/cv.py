# src/cv.py

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

def run_cross_validation(pipeline, X, y, n_splits=5, random_state=42):
    """
    Returns mean/std for AUC, precision, recall, F1 using Stratified K-Fold CV.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    summary = {}
    for k, v in scores.items():
        if k.startswith("test_"):
            metric = k.replace("test_", "")
            summary[metric] = (float(np.mean(v)), float(np.std(v)))

    return summary
