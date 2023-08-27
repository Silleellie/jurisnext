from typing import Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification, BertTokenizer, BertTokenizerFast

from src.model.clustering import ClusterLabelMapper
from src.model.next_title_prediction.ntp_models_interface import NextTitlePredictor


# maybe consider composition rather than multiple inheritance
class NextTitleBert(NextTitlePredictor):

    model_class = BertForSequenceClassification

    def __init__(self,
                 model: BertForSequenceClassification,
                 labels_weights: np.ndarray,
                 tokenizer: Union[BertTokenizer, BertTokenizerFast],
                 cluster_label_mapper: ClusterLabelMapper = None,
                 device: str = "cpu"):

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=torch.optim.AdamW(list(model.parameters()), lr=2e-5),
            cluster_label_mapper=cluster_label_mapper,
            device=device
        )

        self.labels_weights = torch.from_numpy(labels_weights).float().to(device)

    def tokenize(self, sample):

        if self.cluster_label_mapper is not None:

            immediate_next_cluster = self.cluster_label_mapper.get_clusters_from_labels(sample["immediate_next_title"])
            text = ", ".join(sample["input_title_sequence"]) + f"\nNext title cluster: {immediate_next_cluster}"

            output = self.tokenizer(text,
                                    truncation=True)

        else:
            output = self.tokenizer(', '.join(sample["input_title_sequence"]),
                                    truncation=True)

        labels = [self.config.label2id[sample["immediate_next_title"]]]

        return {'input_ids': output['input_ids'],
                'token_type_ids': output['token_type_ids'],
                'attention_mask': output['attention_mask'],
                'labels': labels}

    def prepare_input(self, batch):
        input_dict = {}

        input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        token_type_ids = pad_sequence(batch["token_type_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(batch["attention_mask"], batch_first=True, padding_value=self.tokenizer.pad_token_id)

        input_dict["input_ids"] = input_ids.to(self.device)
        input_dict["token_type_ids"] = token_type_ids.to(self.device)
        input_dict["attention_mask"] = attention_mask.to(self.device)

        if "labels" in batch:
            input_dict["labels"] = batch["labels"].to(self.device).flatten()

        return input_dict

    def train_step(self, batch):

        output = self(**batch)
        truth = batch["labels"]

        loss = torch.nn.functional.cross_entropy(
            output.logits,
            truth,
            weight=self.labels_weights
        )

        return output.logits, loss

    @torch.no_grad()
    def valid_step(self, batch):

        output = self(**batch)
        truth: torch.Tensor = batch["labels"]
        # batch size (batch_size, num_labels) -> (batch_size, 1)
        predictions: torch.Tensor = output.logits.argmax(dim=1)

        val_loss = torch.nn.functional.cross_entropy(
            output.logits,
            truth,
            weight=self.labels_weights
        )

        acc = (predictions == truth).sum()

        return acc.item(), val_loss
