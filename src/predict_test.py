
import os
import pandas as pd
import joblib

from src.config import ID_COL,TEST_PATH, FINAL_MODEL_PATH, TEST_PRED_PATH, THRESHOLD

def main():

    test_path = TEST_PATH
    model_path = FINAL_MODEL_PATH

    df_test = pd.read_csv(test_path)
    clf = joblib.load(model_path)

    X_test = df_test.drop(columns=[ID_COL])

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    out = pd.DataFrame({
        ID_COL: df_test[ID_COL],
        "prediction": pred,
        "probability": proba
    })

    out_path = TEST_PRED_PATH
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
