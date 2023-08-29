import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = str(Path(os.path.join(_THIS_DIR, "..")).resolve())

DATA_DIR = os.path.join(ROOT_PATH, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")
MODELS_DIR = os.path.join(ROOT_PATH, "models")


@dataclass
class ExperimentConfig:
    epochs: int = 100
    batch_size: int = 2
    eval_batch_size: int = 2
    use_cluster_alg: bool = True
    checkpoint: Optional[str] = None
    device: str = "cuda:0"
    random_state: int = 42
