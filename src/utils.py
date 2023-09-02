import os
import random
from contextlib import contextmanager
from typing import Literal

import numpy as np
import torch
import torch.backends.cudnn
import wandb

from src import ExperimentConfig


def seed_everything(seed: int):
    """
    Function which fixes the random state of each library used by this repository with the seed
    specified when invoking `pipeline.py`

    Returns:
        The integer random state set via command line argument

    """

    # seed everything
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    print(f"Random seed set as {seed}")

    return seed


def log_wandb(exp_config: ExperimentConfig, parameters_to_log: dict):
    if exp_config.log_wandb:
        wandb.log(parameters_to_log)


@contextmanager
def init_wandb(exp_name: str, job_type: Literal['data', 'train', 'eval'], log: bool):
    if log:
        with wandb.init(entity="silleellie", project="BD-Next-Title-Prediction", job_type=job_type, group=exp_name):
            yield
    else:
        yield
