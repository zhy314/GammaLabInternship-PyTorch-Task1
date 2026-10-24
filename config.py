"""全局路径、随机种子与训练超参。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

OUTPUT_DIR = ROOT / "outputs"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pt"
HISTORY_CSV = OUTPUT_DIR / "history.csv"
CURVES_PNG = OUTPUT_DIR / "curves.png"
SUBMISSION_CSV = OUTPUT_DIR / "submission.csv"
COMPARE_REPORT = OUTPUT_DIR / "compare_report.txt"
BASELINE_SUBMISSION_CSV = ROOT / "submission.csv"

FEATURES = ["f1", "f2", "f3", "f4"]
N_FEATURES = len(FEATURES)
N_CLASSES = 2
HIDDEN_SIZES = (32, 16)
DROPOUT = 0.1

SEED = 42
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
VAL_RATIO = 0.2
KNN_K = 5
