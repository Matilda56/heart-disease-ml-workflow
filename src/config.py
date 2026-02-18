
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_DIR = SRC_DIR / "models"
REPORT_DIR = SRC_DIR / "reports"

FINAL_MODEL_PATH = MODEL_DIR / "logreg_pipeline.pkl"
TEST_PRED_PATH = REPORT_DIR / "test_predictions.csv"
SUMMARY_PATH = REPORT_DIR / "run_summary.txt"

TARGET_COL = "Heart Disease"
ID_COL = "id"

NUMERIC_COLS = [
    "Age",
    "BP",
    "Cholesterol",
    "Max HR",
    "ST depression",
]

CATEGORICAL_COLS = [
    "Sex",
    "Chest pain type",
    "FBS over 120",
    "EKG results",
    "Exercise angina",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]


LABEL_MAP = {
    "Absence": 0,
    "Presence": 1,
}


RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


THRESHOLD = 0.4
