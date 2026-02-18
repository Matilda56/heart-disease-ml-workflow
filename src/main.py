
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.cv import run_cross_validation
from src.data_validation import basic_data_checks


from src.config import (
    TRAIN_PATH, FINAL_MODEL_PATH,
    TARGET_COL, ID_COL, NUMERIC_COLS, CATEGORICAL_COLS,
    LABEL_MAP, RANDOM_STATE, TEST_SIZE, THRESHOLD,CV_FOLDS, SUMMARY_PATH
)


from src.preprocessing import build_preprocessor
from src.train import get_models, train_model, save_model
from src.evaluate import evaluate_classifier


def main():

    df = pd.read_csv(TRAIN_PATH)

    basic_data_checks(df, required_cols=[TARGET_COL, ID_COL] + NUMERIC_COLS + CATEGORICAL_COLS)

    # Map target to 0/1
    df["target"] = df[TARGET_COL].map(LABEL_MAP)

    X = df.drop(columns=[TARGET_COL, "target", ID_COL])
    y = df["target"]

    # Stratified split for fair validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preprocessor = build_preprocessor(NUMERIC_COLS, CATEGORICAL_COLS)
    models = get_models(random_state=RANDOM_STATE)

    all_val_results = {}

    best_name, best_auc, best_model_obj = None, -1, None

    # 1) Model selection on validation set
    for name, model in models.items():
        clf = train_model(X_train, y_train, preprocessor, model)
        results = evaluate_classifier(clf, X_val, y_val, threshold=THRESHOLD)
        all_val_results[name] = results

        print("\n")
        print(f"Model: {name}")
        print(f"ROC-AUC: {results['roc_auc']:.4f}")
        print(f"Precision: {results['precision']:.4f} | Recall: {results['recall']:.4f} | F1: {results['f1']:.4f}")
        print("Confusion Matrix:\n", results["confusion_matrix"])

        if results["roc_auc"] > best_auc:
            best_auc = results["roc_auc"]
            best_name = name
            best_model_obj = model

    print("\n")
    print(f"Selected model (validation): {best_name} (ROC-AUC={best_auc:.4f})")

    # 2) Cross-validation for robustness (on full training data)
    cv_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", best_model_obj),
    ])

    cv_summary = run_cross_validation(
        cv_pipeline, X, y,
        n_splits=CV_FOLDS,
        random_state=RANDOM_STATE
    )

    print("\n")
    print(f"{CV_FOLDS}-Fold CV Summary (mean ± std):")
    for metric, (mean_v, std_v) in cv_summary.items():
        print(f"{metric}: {mean_v:.4f} ± {std_v:.4f}")

    # 3) Retrain on full training data (train + val)
    final_clf = train_model(X, y, preprocessor, best_model_obj)

    # 4) Save final summary
    model_path = FINAL_MODEL_PATH
    summary_path = SUMMARY_PATH
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w") as f:
        f.write("Model Run Summary\n")

        # (A) Config
        f.write("Config\n")
        f.write(f"Train path: {TRAIN_PATH}\n")
        f.write(f"Test size (val split): {TEST_SIZE}\n")
        f.write(f"Random state: {RANDOM_STATE}\n")
        f.write(f"Threshold: {THRESHOLD}\n\n")

        # (B) Model Selection: validation metrics for ALL models
        f.write("Model Selection (Validation Metrics)\n")
        f.write(f"{'model':<15} {'roc_auc':>10} {'precision':>10} {'recall':>10} {'f1':>10}\n")

        for name, res in sorted(all_val_results.items(), key=lambda x: x[1]["roc_auc"], reverse=True):
            f.write(
                f"{name:<15} "
                f"{res['roc_auc']:>10.4f} "
                f"{res['precision']:>10.4f} "
                f"{res['recall']:>10.4f} "
                f"{res['f1']:>10.4f}\n"
            )
        f.write("\n")

        # (C) Best model details (including confusion matrix)
        f.write("Selected Best Model (by Validation ROC-AUC)\n")
        f.write(f"Selected model: {best_name}\n")
        best_res = all_val_results[best_name]
        f.write(f"Validation ROC-AUC: {best_res['roc_auc']:.4f}\n")
        f.write(f"Precision: {best_res['precision']:.4f}\n")
        f.write(f"Recall: {best_res['recall']:.4f}\n")
        f.write(f"F1: {best_res['f1']:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{best_res['confusion_matrix']}\n\n")

        # (D) K-Fold CV Summary
        f.write(f" {CV_FOLDS}-Fold Cross-Validation (mean ± std)\n")
        for metric, (mean_v, std_v) in cv_summary.items():
            f.write(f"{metric}: {mean_v:.4f} ± {std_v:.4f}\n")
        f.write("\n")

        # (E) Final training & artifact paths
        f.write("Final Training & Artifacts\n")
        f.write("Final model: retrained on full training data (train+val)\n")
        f.write(f"Model saved to: {model_path}\n")
        f.write(f"Summary saved to: {summary_path}\n")

    print(f"Saved run summary to {summary_path}")

    # 5) Save final pipeline
    save_model(final_clf, model_path)
    print(f"Saved final model to {model_path}")


if __name__ == "__main__":
    main()
