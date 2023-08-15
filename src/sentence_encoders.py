from abc import ABC, abstractmethod

import torch
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizerFast


class SentenceEncoder(ABC):

    @abstractmethod
    def get_sentence_embeddings(self, sentences):
        raise NotImplementedError


class SbertSentenceEncoder(SentenceEncoder):

    def __init__(self, model_name='all-MiniLM-L6-v2', model_kwargs=None):

        super().__init__()

        if model_kwargs is None:

            model_kwargs = {}

        self.model = SentenceTransformer(model_name, **model_kwargs)

    def get_sentence_embeddings(self, sentences):
        return self.model.encode(sentences)


class BertSentenceEncoder(SentenceEncoder):

    def __init__(self, model_name='bert-base-uncased', hidden_states_num=4, model_kwargs=None):

        super().__init__()

        if model_kwargs is None:

            model_kwargs = {}

        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True, **model_kwargs)
        self.hidden_states_num = - hidden_states_num

    def get_sentence_embeddings(self, sentences):

        outputs = []

        for sentence in sentences:

            with torch.no_grad():
                output_hidden_states = self.model(**self.tokenizer(sentence, return_tensors='pt')).hidden_states[self.hidden_states_num:]
                output = torch.vstack(output_hidden_states).squeeze().sum(dim=[0, 1]).numpy()
                outputs.append(output)

        return np.vstack(outputs)
