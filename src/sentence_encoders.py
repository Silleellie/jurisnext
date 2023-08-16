from abc import ABC, abstractmethod
from typing import Tuple, Literal

import torch
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizerFast


class SentenceEncoder(ABC):

    @abstractmethod
    def __call__(self, *sentences: Tuple[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEncoder(SentenceEncoder):

    def __init__(self, model_name='all-MiniLM-L6-v2', model_kwargs=None):

        if model_kwargs is None:
            model_kwargs = {}

        self.model = SentenceTransformer(model_name, **model_kwargs)

    def __call__(self, *sentences: Tuple[str]):
        return self.model.encode(*sentences)


class BertSentenceEncoder(SentenceEncoder):

    def __init__(self, model_name='bert-base-uncased',
                 hidden_states_num=4,
                 hidden_states_fusion_strat: Literal["sum", "concat"] = "sum",
                 token_fusion_strat: Literal["sum", "mean"] = "sum",
                 model_kwargs=None):

        if model_kwargs is None:
            model_kwargs = {}

        # keepdim to avoid implicit squeeze when one dim becomes 1
        sum_fusion = lambda x, y: torch.sum(x, dim=y, keepdim=True)
        concat_fusion = lambda x, y: torch.cat(x, dim=y)
        mean_fusion = lambda x, y: torch.mean(x, dim=y, keepdim=True)

        # fusing hidden states involves manipulating dim 0 of the stacked tensor
        hidden_states_available_fusions = {"sum": lambda x: sum_fusion(x, 0), "concat": lambda x: concat_fusion(x, 0)}

        # fusing token embeddings to get a single sentence embedding involves manipluating dim 1 of the stacked tensor
        token_available_fusions = {"sum": lambda x: sum_fusion(x, 1), "mean": lambda x: mean_fusion(x, 1)}


        self.hidden_states_fusion_strat = hidden_states_available_fusions[hidden_states_fusion_strat]
        self.token_fusion_strat = token_available_fusions[token_fusion_strat]

        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True, **model_kwargs)
        self.hidden_states_num = hidden_states_num

    def __call__(self, *sentences: Tuple[str]):

        outputs = []

        for sentence in sentences:

            with torch.no_grad():
                tokenized_sentences = self.tokenizer(sentence, return_tensors='pt', truncation=True)
                output_hidden_states = self.model(**tokenized_sentences).hidden_states[-self.hidden_states_num:]
                output_stacked = torch.vstack(output_hidden_states)

                # TO DO: keep track of dimensions during fusion (at the end dim (1 x hidden_states_optionally_concat)
                sentence_embedding = self.token_fusion_strat(output_stacked)

                hidden_states_fused = self.hidden_states_fusion_strat(sentence_embedding)

                # sentence_embedding = sentence_embedding.squeeze(dim=0)

                outputs.append(sentence_embedding)

        return np.vstack(outputs)
