from __future__ import annotations
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification, BertConfig

from src.model.clustering import ClusterLabelMapper
from src.model.next_title_prediction.ntp_models_abtract import NTPModelHF, NTPConfig


class NTPBertConfig(BertConfig, NTPConfig):

    def __init__(self,
                 labels_weights: list = None,
                 device: str = "cpu",
                 **kwargs):
        BertConfig.__init__(self, **kwargs)
        NTPConfig.__init__(self, device)

        self.labels_weights = labels_weights
        if labels_weights is not None:
            self.labels_weights = torch.from_numpy(np.array(labels_weights)).float().to(device)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):

        labels_weights: Optional[torch.Tensor] = kwargs.pop("labels_weights", None)
        device: Optional[str] = kwargs.pop("device", None)

        if labels_weights is not None:
            config_dict["labels_weights"] = labels_weights

        if device is not None:
            config_dict["device"] = device

        return super().from_dict(config_dict, **kwargs)

    # to make __repr__ work we need to convert the tensor to a json serializable format
    def to_dict(self) -> Dict[str, Any]:
        super_dict = super().to_dict()

        if isinstance(super_dict["labels_weights"], torch.Tensor):
            super_dict["labels_weights"] = super_dict["labels_weights"].tolist()

        return super_dict


# maybe consider composition rather than multiple inheritance
class NTPBert(NTPModelHF):
    model_class = BertForSequenceClassification
    config_class = NTPBertConfig

    def __init__(self,
                 pretrained_model_or_pth: str = 'bert-base-uncased',
                 cluster_label_mapper: ClusterLabelMapper = None,
                 **config_kwargs):

        super().__init__(
            pretrained_model_or_pth=pretrained_model_or_pth,
            cluster_label_mapper=cluster_label_mapper,
            **config_kwargs
        )

    def get_suggested_optimizer(self):
        return torch.optim.AdamW(list(self.model.parameters()), lr=2e-5)

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
        token_type_ids = pad_sequence(batch["token_type_ids"], batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(batch["attention_mask"], batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)

        input_dict["input_ids"] = input_ids.to(self.model.device)
        input_dict["token_type_ids"] = token_type_ids.to(self.model.device)
        input_dict["attention_mask"] = attention_mask.to(self.model.device)

        if "labels" in batch:
            input_dict["labels"] = batch["labels"].to(self.model.device).flatten()

        return input_dict

    def train_step(self, batch):

        output = self(**batch)
        truth = batch["labels"]

        loss = torch.nn.functional.cross_entropy(
            output.logits,
            truth,
            weight=self.config.labels_weights
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
            weight=self.config.labels_weights
        )

        acc = (predictions == truth).sum()

        return acc.item(), val_loss
