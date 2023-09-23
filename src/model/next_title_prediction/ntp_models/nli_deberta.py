import itertools
import random
from math import ceil
from typing import List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DebertaV2ForSequenceClassification, DebertaV2Config, DebertaV2Tokenizer

from src import ExperimentConfig
from src.data.legal_dataset import LegalDataset
from src.model.next_title_prediction.ntp_models_abtract import NTPConfig, NTPModelHF
from src.model.next_title_prediction.ntp_trainer import NTPTrainer


class NTPNliDebertaConfig(DebertaV2Config, NTPConfig):

    def __init__(self,
                 template: str = "Next paragraph title is {}",
                 validation_mini_batch_size: int = 16,
                 all_unique_labels: List[str] = None,
                 device: str = "cpu",
                 **kwargs):
        DebertaV2Config.__init__(self, **kwargs)
        NTPConfig.__init__(self, device)

        self.validation_mini_batch_size = validation_mini_batch_size
        self.template = template
        self.all_unique_labels = all_unique_labels

        if self.all_unique_labels is None:
            self.all_unique_labels = []


class NTPNliDeberta(NTPModelHF):

    model_class = DebertaV2ForSequenceClassification
    tokenizer_class = DebertaV2Tokenizer
    config_class = NTPNliDebertaConfig
    default_checkpoint = 'cross-encoder/nli-deberta-v3-xsmall'

    def __init__(self,
                 pretrained_model_or_pth: str = default_checkpoint,
                 **kwargs):

        super().__init__(
            pretrained_model_or_pth=pretrained_model_or_pth,
            **kwargs
        )

    def get_suggested_optimizer(self):
        return torch.optim.AdamW(list(self.model.parameters()), lr=2e-5)

    def tokenize(self, sample):

        next_candidate_titles = np.array(self.config.all_unique_labels)
        text = ", ".join(sample["input_title_sequence"])

        label = sample["immediate_next_title"]
        label_ent: int = self.config.label2id["entailment"]
        label_contr: int = self.config.label2id["contradiction"]

        if self.training:

            wrong_label = random.choice(next_candidate_titles[next_candidate_titles != label])
            encoded_sequence = self.tokenizer([[text, self.config.template.format(label)],
                                               [text, self.config.template.format(wrong_label)]],
                                              truncation=True)

            encoded_sequence["labels"] = [label_ent, label_contr]

        else:

            encoded_sequence = self.tokenizer([[text, self.config.template.format(candidate_title)]
                                               for candidate_title in next_candidate_titles],
                                              return_tensors='pt',
                                              padding=True,
                                              truncation=True)

            encoded_sequence["labels"] = torch.full(size=next_candidate_titles.shape, fill_value=label_contr)

            next_possible_titles_correct_idx = next_candidate_titles == label
            encoded_sequence["labels"][next_possible_titles_correct_idx] = label_ent

            encoded_sequence["labels"] = encoded_sequence["labels"].tolist()
            encoded_sequence["text_labels"] = [candidate_title for candidate_title in next_candidate_titles]

        return encoded_sequence

    def prepare_input(self, batch):
        input_dict = {}

        if self.training:
            # for each batch entry, we have 2 inputs: correct label (entailment), wrong label (contradiction).
            # we need to flatten and pad accordingly
            flat_input_ids = itertools.chain.from_iterable(batch["input_ids"])
            flat_token_type_ids = itertools.chain.from_iterable(batch["token_type_ids"])
            flat_attention_mask = itertools.chain.from_iterable(batch["attention_mask"])

            input_ids = pad_sequence(flat_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            token_type_ids = pad_sequence(flat_token_type_ids, batch_first=True,
                                          padding_value=self.tokenizer.pad_token_id)
            attention_mask = pad_sequence(flat_attention_mask, batch_first=True,
                                          padding_value=self.tokenizer.pad_token_id)

            input_dict["input_ids"] = input_ids.to(self.model.device)
            input_dict["token_type_ids"] = token_type_ids.to(self.model.device)
            input_dict["attention_mask"] = attention_mask.to(self.model.device)

        else:

            input_dict["input_ids"] = [x.to(self.model.device) for x in batch["input_ids"]]
            input_dict["token_type_ids"] = [x.to(self.model.device) for x in batch["token_type_ids"]]
            input_dict["attention_mask"] = [x.to(self.model.device) for x in batch["attention_mask"]]

        if "labels" in batch:

            if self.training:
                flat_labels = batch["labels"].flatten()  # already a tensor
                input_dict["labels"] = flat_labels.to(self.model.device)
            else:
                input_dict["labels"] = [x.to(self.model.device).long() for x in batch["labels"]]
                input_dict["text_labels"] = batch["text_labels"]

        return input_dict

    def train_step(self, batch):
        output = self(**batch)

        entail_contradiction_logits = output.logits[:,
                                      [self.config.label2id["contradiction"], self.config.label2id["entailment"]]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1]

        loss = torch.nn.functional.binary_cross_entropy(
            prob_label_is_true,
            batch["labels"].float()
        )

        return loss

    @torch.no_grad()
    def valid_step(self, batch):

        text_labels = batch.pop("text_labels")
        mini_batch_size = self.config.validation_mini_batch_size

        val_loss = 0
        label_contr = self.config.label2id["contradiction"]
        label_ent = self.config.label2id["entailment"]

        predictions = []
        truths = []
        # we cycle through each sample of the batch, each sample has associated
        # (num_candidate_labels x pad_dim) input_ids, labels, attention_mask, etc.
        for i in range(len(batch["input_ids"])):

            output_logits = []
            # ceil to not drop the last batch
            max_j = ceil(batch['input_ids'][i].shape[0] / mini_batch_size)

            for j in range(max_j):
                mini_batch = {k: v[i][j * mini_batch_size:(j + 1) * mini_batch_size] for k, v in batch.items()}

                output = self(**mini_batch)
                output_logits.append(output.logits)

            output_logits = torch.vstack(output_logits)
            entail_contradiction_logits = output_logits[:, [label_contr, label_ent]]
            probs = entail_contradiction_logits.softmax(dim=1)

            prob_label_is_true = probs[:, 1]
            truth: torch.Tensor = batch["labels"][i]

            loss = torch.nn.functional.binary_cross_entropy(
                prob_label_is_true,
                truth.float()
            )

            truth: torch.Tensor = truth

            # get index of label which is entailment in prediction tensor
            prediction_index: torch.Tensor = prob_label_is_true.argmax(0).item()
            prediction_text = text_labels[i][prediction_index]

            # get index of label which is entailment in truth tensor
            truth_index = (truth == self.config.label2id["entailment"]).nonzero().item()
            truth_text = text_labels[i][truth_index]

            # increment accuracy if they match
            predictions.append(prediction_text)
            truths.append(truth_text)

            val_loss += loss

        return predictions, truths, val_loss


def nli_deberta_main(exp_config: ExperimentConfig):

    n_epochs = exp_config.epochs
    batch_size = exp_config.train_batch_size
    eval_batch_size = exp_config.eval_batch_size
    device = exp_config.device

    checkpoint = 'cross-encoder/nli-deberta-v3-xsmall'
    if exp_config.checkpoint is not None:
        checkpoint = exp_config.checkpoint

    ds = LegalDataset.load_dataset(exp_config)
    dataset = ds.get_hf_datasets()
    all_unique_labels = ds.all_unique_labels
    sampling_fn = ds.perform_sampling

    train = dataset["train"]
    val = dataset["validation"]

    ntp_model = NTPNliDeberta(
        pretrained_model_or_pth=checkpoint,
        all_unique_labels=list(all_unique_labels),
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
