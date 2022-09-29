from pathlib import Path

PROJECT_NAME = Path.cwd().name
DESCRIPTION = """
Distilling a model through an API.
"""

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
SAVED_MODELS_DIR = PROJECT_DIR / "save_pretrained"

# BRANCHES
MAIN = "main"

# SCRIPT PATHS
MAIN_PATH = PROJECT_DIR / "main.py"

# GENERAL
IMDB = "imdb"
AUTO = "auto"

# WANDB
WANDB = "wandb"
WANDB_PROJECT = f"WANDB_{PROJECT_NAME}"

# SUFFIXES
CSV = '.csv'
JSON = '.json'
TXT = '.txt'
JPG = '.jpg'
PNG = '.png'

# SPLITS
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
UNLABELED = "unlabeled"
ALL = "all"
SPLITS = [TRAIN, VALIDATION, TEST, UNLABELED]

# MODEL TYPES
CLS = "cls"  # Classification
QA = "qa"  # Question Answering
TCLS = "tcls"  # Token Classification
ALL_MODEL_TYPES = [CLS, QA, TCLS]

# TRAINER TYPES
STANDARD = "standard"
CUSTOM = "custom"
ALL_TRAINER_TYPES = [STANDARD, CUSTOM]

# METRICS
AGREEMENT = "agreement"
ACCURACY = "accuracy"
NOISE = "iteration_noise"
F1 = "f1"
MACRO = "macro"
MACRO_F1 = "macro-f1"
MATTHEWS_CORRELATION = "matthews_correlation"

# FEATURES
TEXT = "text"
NER = "ner"
LABELS = "labels"
INPUT_IDS = "input_ids"
TRAIN_SAMPLES = "train_samples"
EVAL = "eval"
EVAL_SAMPLES = "eval_samples"
