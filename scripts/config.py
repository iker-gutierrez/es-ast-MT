import torch
from pathlib import Path

BASE_DIR = Path("./workspace")

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "model_ft"

BASELINE_PRED_DIR = BASE_DIR / "predictions_baseline"

FT_PRED_DIR = BASE_DIR / "predictions_ft"

HF_DATASET_DIR = DATA_DIR / "hf_dataset"

MODEL_NAME = "facebook/nllb-200-distilled-600M"

SRC_LANG = "spa_Latn"
TGT_LANG = "ast_Latn"

MAX_LENGTH = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)