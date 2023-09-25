import os
import random
from contextlib import contextmanager
from typing import Literal

import numpy as np
import torch
import torch.backends.cudnn
import wandb

from src import ExperimentConfig


def add_cluster_column(ntp_model, sampled_ds, dataset_split: str, batch_size: int):

    preprocessed_ds_supp = sampled_ds.map(ntp_model.prediction_supporter.tokenize,
                                          remove_columns=sampled_ds.column_names,
                                          load_from_cache_file=False,
                                          keep_in_memory=True,
                                          desc=f"Tokenizing {dataset_split} for prediction supporter")
    preprocessed_ds_supp.set_format("torch")

    predicted_clusters = preprocessed_ds_supp.map(ntp_model.prepare_batch_pred_supp,
                                                  batched=True,
                                                  batch_size=batch_size,
                                                  remove_columns=preprocessed_ds_supp.column_names,
                                                  load_from_cache_file=False,
                                                  keep_in_memory=True,
                                                  desc="Computing cluster prediction")

    sampled_ds = sampled_ds.add_column("predicted_cluster", predicted_clusters["clusters"].numpy())

    return sampled_ds


def seed_everything(seed: int, print_seed: bool = True):
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

    if print_seed:
        print(f"Random seed set as {seed}")

    return seed


def log_wandb(exp_config: ExperimentConfig, parameters_to_log: dict):
    if exp_config.log_wandb:
        wandb.log(parameters_to_log)


@contextmanager
def init_wandb(exp_name: str, job_type: Literal['data', 'train', 'eval'], log: bool):
    if log:
        with wandb.init(project="BD-Next-Title-Prediction", job_type=job_type, group=exp_name):
            yield
    else:
        yield
