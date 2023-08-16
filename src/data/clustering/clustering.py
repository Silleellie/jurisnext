from abc import ABC, abstractmethod

from collections import defaultdict

import numpy_indexed as npi

from sentence_transformers import util
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

from src.sentence_encoders import SentenceEncoder


class ClusterLearner(ABC):

    def __init__(self, sentences, sentence_encoder: SentenceEncoder, alg_args=None):

        self.sentence_encoder = sentence_encoder
        self.cluster_idx_to_sentence = {}
        self.clustering_alg = self.learn_clusters(sentences, alg_args)

    @abstractmethod
    def learn_clusters(self, sentences, alg_args=None):
        raise NotImplementedError

    @abstractmethod
    def get_sentence_mapping(self, sentences):
        raise NotImplementedError


class KMeansLearner(ClusterLearner):

    def __init__(self, sentences, sentence_encoder: SentenceEncoder, alg_args=None):

        super().__init__(sentences, sentence_encoder, alg_args)

    def learn_clusters(self, sentences, kmeans_args=None):

        embeddings = self.sentence_encoder.get_sentence_embeddings(sentences)

        if kmeans_args is None:

            kmeans_args = {
                'n_clusters': 50,
                'random_state': 42,
                'init': 'k-means++',
                'n_init': 'auto'
            }

        km = KMeans(**kmeans_args).fit(embeddings)

        self.cluster_idx_to_sentence = {}

        for i, center in enumerate(km.cluster_centers_):

            most_sim_idx = util.cos_sim(center, embeddings).argmax(1)
            self.cluster_idx_to_sentence[i] = sentences[most_sim_idx]

        return km

    def get_sentence_mapping(self, sentences):

        to_sentence = {}
        sentence_cluster_to_others = defaultdict(list)

        embeddings = self.sentence_encoder.get_sentence_embeddings(sentences)

        for i, embedding in enumerate(embeddings):

            most_sim_cluster_idx = util.cos_sim(embedding, self.clustering_alg.cluster_centers_).argmax(1).item()
            predict = self.cluster_idx_to_sentence[most_sim_cluster_idx]
            to_sentence[sentences[i]] = predict
            sentence_cluster_to_others[predict].append(sentences[i])

        return to_sentence, sentence_cluster_to_others


class KMedoidsLearner(ClusterLearner):

    def __init__(self, sentences, sentence_encoder: SentenceEncoder, alg_args=None):

        super().__init__(sentences, sentence_encoder, alg_args)

    def learn_clusters(self, sentences, alg_args=None):

        embeddings = self.sentence_encoder.get_sentence_embeddings(sentences)

        if alg_args is None:

            alg_args = {
                'n_clusters': 50,
                'random_state': 42,
                'metric': 'cosine',
                'init': 'k-medoids++'
            }

        return KMedoids(**alg_args).fit(embeddings)

    def get_sentence_mapping(self, sentences):

        to_sentence = {}
        sentence_cluster_to_others = defaultdict(list)

        embeddings = self.sentence_encoder.get_sentence_embeddings(sentences)

        for i, embedding in enumerate(embeddings):
            # predict best cluster for embedding
            predicted_cluster_idx = self.clustering_alg.predict([embedding])
            # find sentence associated to predicted cluster
            cluster_sentence_idx = npi.contains(
                [self.clustering_alg.cluster_centers_[predicted_cluster_idx][0]], embeddings).nonzero()[0][0]
            predict = sentences[cluster_sentence_idx]
            to_sentence[sentences[i]] = predict
            sentence_cluster_to_others[predict].append(sentences[i])

        return to_sentence, sentence_cluster_to_others


if __name__ == "__main__":

    import pickle
    import pandas as pd
    import os

    from src import RAW_DATA_DIR, PROCESSED_DATA_DIR
    from src.sentence_encoders import SbertSentenceEncoder

    with open(os.path.join(RAW_DATA_DIR, "pre-processed_representations.pkl"), "rb") as f:
        data: pd.DataFrame = pickle.load(f)

    all_labels = data['concept:name']
    all_unique_labels = pd.unique(data['concept:name'])

    kmeans = KMeansLearner(all_labels, SbertSentenceEncoder())
    sentence_mapping, sentence_cluster_to_others = kmeans.get_sentence_mapping(all_unique_labels)
    data = data.replace({'concept:name': sentence_mapping})

    data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'clustered_dataset.csv'))
