import os
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = str(Path(os.path.join(THIS_DIR, "..")).resolve())

DATA_DIR = os.path.join(ROOT_PATH, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
MODELS_DIR = os.path.join(ROOT_PATH, "models")

RANDOM_STATE = 42
