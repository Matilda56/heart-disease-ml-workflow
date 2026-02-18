
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .preprocessing import build_preprocessor

def get_models(random_state=42):
    models = {
        "logreg": LogisticRegression(max_iter=1000, n_jobs=None),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1
        ),
    }
    return models

def train_model(X_train, y_train, preprocessor, model):
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])
    clf.fit(X_train, y_train)
    return clf

def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
