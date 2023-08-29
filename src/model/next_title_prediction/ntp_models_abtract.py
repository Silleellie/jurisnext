import os
import pickle
from abc import abstractmethod, ABC
from typing import Union

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer

from src.model.clustering import ClusterLabelMapper


class NTPConfig:

    def __init__(self, device: str = "cpu"):
        self.device = device


# interface for all sequence classification models
class NTPModel:

    model_class = None

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 cluster_label_mapper: ClusterLabelMapper = None):

        self.model = model
        self.tokenizer = tokenizer
        self.cluster_label_mapper = cluster_label_mapper

        self.model.to(self.model.config.device)

    @abstractmethod
    def get_suggested_optimizer(self):
        raise NotImplementedError

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

    @property
    def config(self):
        return self.model.config

    def train(self, mode: bool = True):
        return self.model.train(mode)

    def eval(self):
        return self.model.eval()

    @property
    def training(self):
        return self.model.training

    def _train_clusters(self, unique_train_labels: np.ndarray, all_labels: np.ndarray):
        # fit the cluster label mapper with train labels and all labels which should be clustered (both are unique)
        self.cluster_label_mapper.fit(unique_train_labels, all_labels)

    def save(self, save_path):

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        if self.cluster_label_mapper is not None:
            with open(os.path.join(save_path, 'cluster_label_mapper.pkl'), "wb") as f:
                pickle.dump(self.cluster_label_mapper, f)

    @classmethod
    def load(cls, save_path):

        model = cls.model_class.from_pretrained(
            pretrained_model_name_or_path=save_path
        )

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=save_path
        )

        cluster_label_mapper_path = os.path.join(save_path, 'cluster_label_mapper.pkl')

        cluster_label_mapper = None
        if os.path.isfile(cluster_label_mapper_path):
            with open(cluster_label_mapper_path, "rb") as f:
                cluster_label_mapper = pickle.load(f)

        new_inst = cls(
            model=model,
            tokenizer=tokenizer,
            cluster_label_mapper=cluster_label_mapper
        )

        return new_inst

    @staticmethod
    def compute_metrics(predictions, truths):

        predictions = np.array(predictions)
        truths = np.array(truths)

        metrics_results = {}

        precision_micro = precision_score(truths, predictions, average='micro')
        precision_weighted = precision_score(truths, predictions, average='weighted')

        recall_micro = recall_score(truths, predictions, average='micro')
        recall_weighted = recall_score(truths, predictions, average='weighted')

        f1_micro = f1_score(truths, predictions, average='micro')
        f1_weighted = f1_score(truths, predictions, average='weighted')

        accuracy = accuracy_score(truths, predictions)

        metrics_results['Precision (micro)'] = precision_micro
        metrics_results['Precision (weighted)'] = precision_weighted

        metrics_results['Recall (micro)'] = recall_micro
        metrics_results['Recall (weighted)'] = recall_weighted

        metrics_results['F1 (micro)'] = f1_micro
        metrics_results['F1 (weighted)'] = f1_weighted

        metrics_results['Accuracy'] = accuracy

        return metrics_results

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class NTPModelHF(ABC, NTPModel):

    config_class = NTPConfig

    def __init__(self,
                 pretrained_model_or_pth: str,
                 cluster_label_mapper: ClusterLabelMapper = None,
                 **config_kwargs):

        self.model_class.config_class = self.config_class
        model = self.model_class.from_pretrained(pretrained_model_or_pth, **config_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_or_pth)

        super().__init__(model, tokenizer, cluster_label_mapper)

    @classmethod
    def load(cls, save_path):
        cls.model_class.config_class = cls.config_class

        cluster_label_mapper_path = os.path.join(save_path, 'cluster_label_mapper.pkl')

        cluster_label_mapper = None
        if os.path.isfile(cluster_label_mapper_path):
            with open(cluster_label_mapper_path, "rb") as f:
                cluster_label_mapper = pickle.load(f)

        return cls(
            pretrained_model_or_pth=save_path,
            cluster_label_mapper=cluster_label_mapper
        )
