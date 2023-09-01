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
REPORTS_DIR = os.path.join(ROOT_PATH, "reports")
METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")


@dataclass
class ExperimentConfig:

    # these 3 are set in pipeline.py depending on cmd parameters passed
    model: Optional[str]
    checkpoint: Optional[str]
    exp_name: Optional[str]

    epochs: int = 100
    train_batch_size: int = 2
    eval_batch_size: int = 2
    random_seed: int = 42
    use_clusters: bool = False
    log_wandb: bool = False
    n_test_set: int = 10
    device: str = "cuda:0"
