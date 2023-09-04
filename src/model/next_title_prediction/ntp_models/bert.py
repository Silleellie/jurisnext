from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification, BertConfig

from src import ExperimentConfig
from src.data.legal_dataset import LegalDataset
from src.model.clustering import ClusterLabelMapper, KMeansAlg
from src.model.next_title_prediction.ntp_models_abtract import NTPModelHF, NTPConfig
from src.model.next_title_prediction.ntp_trainer import NTPTrainer
from src.model.sentence_encoders import SentenceTransformerEncoder


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
    default_checkpoint = 'bert-base-uncased'

    def __init__(self,
                 pretrained_model_or_pth: str = default_checkpoint,
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

        return loss

    @torch.no_grad()
    def valid_step(self, batch):

        output = self(**batch)
        truth = batch["labels"]
        # batch size (batch_size, num_labels) -> (batch_size, 1)
        predictions = output.logits.argmax(dim=1)

        val_loss = torch.nn.functional.cross_entropy(
            output.logits,
            truth,
            weight=self.config.labels_weights
        )

        predictions = [self.config.id2label[x.cpu().item()] for x in predictions]
        truth = [self.config.id2label[x.cpu().item()] for x in truth]

        return predictions, truth, val_loss


def bert_main(exp_config: ExperimentConfig):

    n_epochs = exp_config.epochs
    batch_size = exp_config.train_batch_size
    eval_batch_size = exp_config.eval_batch_size
    device = exp_config.device
    use_cluster_alg = exp_config.use_clusters

    checkpoint = "bert-base-uncased"
    if exp_config.checkpoint is not None:
        checkpoint = exp_config.checkpoint

    random_state = exp_config.random_seed

    ds = LegalDataset.load_dataset()
    dataset = ds.get_hf_datasets()
    all_unique_labels = ds.all_unique_labels

    cluster_label = None

    if use_cluster_alg:
        clus_alg = KMeansAlg(
            n_clusters=50,
            random_state=random_state,
            init="k-means++",
            n_init="auto"
        )

        sent_encoder = SentenceTransformerEncoder(
            device=device,
        )

        cluster_label = ClusterLabelMapper(sent_encoder, clus_alg)

    train = dataset["train"]
    val = dataset["validation"]

    all_train_labels_occurrences = [y for x in train for y in x['title_sequence']]
    # "smoothing" so that a weight can be calculated for labels which do not appear in the
    # train set
    all_train_labels_occurrences.extend(all_unique_labels)

    labels_weights = compute_class_weight(class_weight='balanced',
                                          classes=all_unique_labels,
                                          y=all_train_labels_occurrences)

    ntp_model = NTPBert(
        checkpoint,
        cluster_label_mapper=cluster_label,

        problem_type="single_label_classification",
        num_labels=len(all_unique_labels),
        label2id={x: i for i, x in enumerate(all_unique_labels)},
        id2label={i: x for i, x in enumerate(all_unique_labels)},

        labels_weights=list(labels_weights),
        device=device
    )

    trainer = NTPTrainer(
        ntp_model=ntp_model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        all_labels=all_unique_labels,
        eval_batch_size=eval_batch_size,
        output_name=exp_config.exp_name,
        log_wandb=exp_config.log_wandb
    )

    trainer.train(train, val)

    return trainer.output_name
