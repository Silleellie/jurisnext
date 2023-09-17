import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, ClassVar, Literal

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

    model: Optional[str] = None
    checkpoint: Optional[str] = None
    exp_name: Optional[str] = None
    pipeline_phases: List[str] = None
    epochs: int = 100
    train_batch_size: int = 2
    eval_batch_size: int = 2
    random_seed: int = 42
    monitor_strategy: Literal['loss', 'metric'] = 'metric'
    use_clusters: bool = False
    log_wandb: bool = False
    n_test_set: int = 10
    ngram_label: Optional[int] = None
    seq_sampling_strategy: Literal['random', 'augment'] = "random"
    seq_sampling_start_strategy: Literal['beginning', 'random'] = "beginning"
    clean_stopwords_kwds: bool = False
    t5_keyword_min_occ: Optional[int] = None
    t5_tasks: List[str] = None
    device: str = "cuda:0"
