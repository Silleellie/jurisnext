import os
import pickle
from abc import abstractmethod

import numpy as np
import torch

from src.model.clustering import ClusterLabelMapper


# interface for all sequence classification models
class NextTitlePredictor:

    model_class = None

    def __init__(self, model, tokenizer, optimizer, cluster_label_mapper: ClusterLabelMapper = None, device: str = "cuda:0"):

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.cluster_label_mapper = cluster_label_mapper
        self.device = device
        self.model.to(device)

    @property
    def config(self):
        return self.model.config

    def eval(self):
        return self.model.eval()

    def train(self):
        return self.model.train()

    @property
    def training(self):
        return self.model.training

    @abstractmethod
    def tokenize(self, sample):
        raise NotImplementedError

    @abstractmethod
    def prepare_input(self, batch):
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError

    @abstractmethod
    def valid_step(self, batch):
        raise NotImplementedError

    def _train_clusters(self, unique_train_labels: np.ndarray, all_labels: np.ndarray):
        # fit the cluster label mapper with train labels and all labels which should be clustered (both are unique)
        self.cluster_label_mapper.fit(unique_train_labels, all_labels)

    def save_pretrained(self, save_path):

        self.model.save_pretrained(save_path)

        with open(os.path.join(save_path, 'cluster_label_mapper.pkl'), "wb") as f:
            pickle.dump(self.cluster_label_mapper, f)

        torch.save(self.optimizer.state_dict(), os.path.join(save_path, 'optimizer.pt'))

    @classmethod
    def from_pretrained(cls, save_path, **kwargs):

        model = cls.model_class.from_pretrained(
            pretrained_model_name_or_path=save_path
        )

        with open(os.path.join(save_path, 'cluster_label_mapper.pkl'), "rb") as f:
            cluster_label_mapper = pickle.load(f)

        new_inst = cls(
            model=model,
            cluster_label_mapper=cluster_label_mapper,
            **kwargs
        )

        new_inst.optimizer.load_state_dict(torch.load(os.path.join(save_path, 'optimizer.pt')))

        return new_inst

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
