from pathlib import Path

PROJECT_NAME = "IMDB_Multiclass"

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"

TRAIN = 'train'
DEV = 'dev'
TEST = 'test'

CSV = '.csv'
TXT = '.txt'
YAML = '.yaml'

UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"

TEXT = "text"
INPUT_IDS = "input_ids"
LABEL = "label"
TEXT_LEN = "text_length"

EPOCH = "epoch"
ITERATION = "iteration"

MODEL_WEIGHTS = ".model_weights"
SUBMISSION = "submission"

ACC_THRESHOLD = 0.4
