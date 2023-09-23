from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification, BertConfig

from src import ExperimentConfig
from src.data.legal_dataset import LegalDataset
from src.model.next_title_prediction.ntp_models_abtract import NTPModelHF, NTPConfig
from src.model.next_title_prediction.ntp_trainer import NTPTrainer


class NTPBertConfig(BertConfig, NTPConfig):

    def __init__(self,
                 device: str = "cpu",
                 **kwargs):
        BertConfig.__init__(self, **kwargs)
        NTPConfig.__init__(self, device)


# maybe consider composition rather than multiple inheritance
class NTPBert(NTPModelHF):
    model_class = BertForSequenceClassification
    config_class = NTPBertConfig
    default_checkpoint = 'bert-base-uncased'

    def __init__(self,
                 pretrained_model_or_pth: str = default_checkpoint,
                 **config_kwargs):

        super().__init__(
            pretrained_model_or_pth=pretrained_model_or_pth,
            **config_kwargs
        )

    def get_suggested_optimizer(self):
        return torch.optim.AdamW(list(self.model.parameters()), lr=2e-5)

    def tokenize(self, sample):

        output = self.tokenizer(', '.join(sample["input_title_sequence"]), truncation=True)
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
        return output.loss

    @torch.no_grad()
    def valid_step(self, batch):
        output = self(**batch)
        truth = batch["labels"]
        # batch size (batch_size, num_labels) -> (batch_size, 1)
        predictions = output.logits.argmax(dim=1)

        predictions = [self.config.id2label[x.cpu().item()] for x in predictions]
        truth = [self.config.id2label[x.cpu().item()] for x in truth]

        return predictions, truth, output.loss


def bert_main(exp_config: ExperimentConfig):

    n_epochs = exp_config.epochs
    batch_size = exp_config.train_batch_size
    eval_batch_size = exp_config.eval_batch_size
    device = exp_config.device

    checkpoint = "bert-base-uncased"
    if exp_config.checkpoint is not None:
        checkpoint = exp_config.checkpoint

    ds = LegalDataset.load_dataset(exp_config)
    dataset = ds.get_hf_datasets()
    all_unique_labels = ds.all_unique_labels
    sampling_fn = ds.perform_sampling

    train = dataset["train"]
    val = dataset["validation"]

    ntp_model = NTPBert(
        checkpoint,

        problem_type="single_label_classification",
        num_labels=len(all_unique_labels),
        label2id={x: i for i, x in enumerate(all_unique_labels)},
        id2label={i: x for i, x in enumerate(all_unique_labels)},

        device=device
    )

    trainer = NTPTrainer(
        ntp_model=ntp_model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        all_labels=all_unique_labels,
        eval_batch_size=eval_batch_size,
        output_name=exp_config.exp_name,
        log_wandb=exp_config.log_wandb,
        train_sampling_fn=sampling_fn,
        monitor_strategy=exp_config.monitor_strategy
    )

    trainer.train(train, val)

    return trainer.output_name
