from __future__ import annotations
from abc import ABC, abstractmethod

import os
import pickle
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

from src import RAW_DATA_DIR
from src.model.sentence_encoders import SentenceEncoder, BertSentenceEncoder


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


class ClusterLabelMapper:

    def __init__(self, sentence_encoder: SentenceEncoder, cluster_alg: ClusterAlg):
        self.sentence_encoder = sentence_encoder
        self.clustering_alg = cluster_alg

        # ["summary", "cost", "eu law reference", ...]
        self.labels_arr: Optional[np.ndarray] = None

        # [0, 1, 0, ...]
        self.cluster_arr: Optional[np.ndarray] = None

    def fit(self, train_sentences: np.ndarray[str], all_sentences: np.ndarray[str]) -> ClusterLabelMapper:
        encoded_train_sentences = self.sentence_encoder(*train_sentences,
                                                        desc="Encoding TRAIN labels for clustering...")
        self.clustering_alg.fit(encoded_train_sentences)

        encoded_pred_sentences = self.sentence_encoder(*all_sentences,
                                                       desc="Encoding ALL labels for clustering...")
        pred_clusters = self.clustering_alg.predict(encoded_pred_sentences)

        self.labels_arr = all_sentences
        self.cluster_arr = pred_clusters

        return self

    def get_clusters_from_labels(self, labels: Union[str, Iterable[str]]) -> np.ndarray[int]:
        # check only one of the two, no need to check both
        assert self.labels_arr is not None, "call fit_predict method first!"

        bool_mask = np.where(self.labels_arr == labels)

        return self.cluster_arr[bool_mask]

    def get_labels_from_cluster(self, clusters: int) -> np.ndarray[str]:
        # check only one of the two, no need to check both
        assert self.labels_arr is not None, "call fit_predict method first!"

        bool_mask = np.where(self.cluster_arr == clusters)

        return self.labels_arr[bool_mask]


if __name__ == "__main__":
    with open(os.path.join(RAW_DATA_DIR, "pre-processed_representations.pkl"), "rb") as f:
        data: pd.DataFrame = pickle.load(f)

    all_labels = data['concept:name']
    all_unique_labels = pd.unique(data['concept:name'])

    # TO DO: TRAIN ONLY ON TRAIN SET
    clus_alg = KMeansAlg(
        n_clusters=50,
        random_state=42,
        init="k-means++",
        n_init="auto"
    )
    kmeans = ClusterLabelMapper(
        sentence_encoder=BertSentenceEncoder(model_name="nlpaueb/legal-bert-base-uncased",
                                             token_fusion_strat="mean",
                                             hidden_states_fusion_strat="concat"),
        cluster_alg=clus_alg
    )

    # MOCK TRAIN
    mock_train_unique_labels = all_unique_labels[:500]
    mock_all_unique_labels = all_unique_labels[:700]

    kmeans.fit(train_sentences=mock_train_unique_labels,
               all_sentences=mock_all_unique_labels[:600])

    # data["concept:name"] = labels_converted_to_clusters

    # print("we")
    # data = data.replace({'concept:name': sentence_mapping})
    #
    # data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'clustered_dataset.csv'))
