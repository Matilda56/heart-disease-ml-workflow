
import pandas as pd

def basic_data_checks(df: pd.DataFrame, required_cols=None) -> None:
    print("Basic Data Checks")
    print(f"Shape: {df.shape}")
    print("\nMissing values (top 10):")
    missing = df.isna().sum().sort_values(ascending=False).head(10)
    print(missing[missing > 0] if (missing > 0).any() else "No missing values detected.")

    dup = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup}")

    if required_cols:
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
