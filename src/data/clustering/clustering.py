from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Tuple
import os
import pickle

import numpy as np
import torch
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

from src import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.sentence_encoders import SentenceEncoder, BertSentenceEncoder


class ClusterAlg(ABC):

    @abstractmethod
    def fit(self, sentences) -> ClusterAlg:
        raise NotImplementedError

    @abstractmethod
    def predict(self, sentences_encoded: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class KMeansAlg(ClusterAlg):

    def __init__(self, **kwargs):
        self.alg = KMeans(**kwargs)

    def fit(self, sentences_encoded: np.ndarray) -> ClusterAlg:
        self.alg = self.alg.fit(sentences_encoded)
        return self

    def predict(self, sentences_encoded: np.ndarray) -> np.ndarray:
        return self.alg.predict(sentences_encoded)


class KMedoidsAlg(ClusterAlg):
    def __init__(self, **kwargs):
        self.alg = KMedoids(**kwargs)

    def fit(self, sentences_encoded: np.ndarray) -> ClusterAlg:
        self.alg = self.alg.fit(sentences_encoded)
        return self

    def predict(self, sentences_encoded: np.ndarray) -> np.ndarray:
        return self.alg.predict(sentences_encoded)


class ClusterLabel:

    def __init__(self, sentences: np.ndarray, sentence_encoder: SentenceEncoder, cluster_alg: ClusterAlg):

        self.sentence_encoder = sentence_encoder

        encoded_fit_sentences = sentence_encoder(*sentences)
        self.clustering_alg = cluster_alg.fit(encoded_fit_sentences)

    def __call__(self, *sentences: str) -> torch.IntTensor:

        sentences_encoded = self.sentence_encoder(*sentences)
        cluster_idxs = self.clustering_alg.predict(sentences_encoded)

        return torch.IntTensor(cluster_idxs)


if __name__ == "__main__":

    with open(os.path.join(RAW_DATA_DIR, "pre-processed_representations.pkl"), "rb") as f:
        data: pd.DataFrame = pickle.load(f)

    all_labels = data['concept:name']
    all_unique_labels = pd.unique(data['concept:name'])

    clus_alg = KMeansAlg(
        n_clusters=50,
        random_state=42,
        init="k-means++",
        n_init="auto"
    )
    kmeans = ClusterLabel(
        sentences=all_unique_labels,
        sentence_encoder=BertSentenceEncoder(hidden_states_fusion_strat="concat"),
        cluster_alg=clus_alg
    )

    print("we")
    data = data.replace({'concept:name': sentence_mapping})

    data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'clustered_dataset.csv'))
