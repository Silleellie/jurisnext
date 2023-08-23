from abc import abstractmethod

import numpy as np

from src.model.clustering import ClusterLabelMapper


# interface for all sequence classification models
class SeqClassification:

    def __init__(self, tokenizer, optimizer, cluster_label_mapper: ClusterLabelMapper = None):

        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.cluster_label_mapper = cluster_label_mapper

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
