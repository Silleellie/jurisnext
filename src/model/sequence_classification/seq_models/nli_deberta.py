import itertools
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DebertaV2ForSequenceClassification

from src.model.sequence_classification.seq_models_interface import SeqClassification


class FineTunedNliDeberta(DebertaV2ForSequenceClassification, SeqClassification):

    def __init__(self, config, labels_weights: np.ndarray, all_unique_labels: np.ndarray, tokenizer):
        DebertaV2ForSequenceClassification.__init__(self, config)

        SeqClassification.__init__(
            self,
            tokenizer=tokenizer,
            optimizer=torch.optim.AdamW(list(self.parameters()), lr=2e-5)
        )

        self.all_unique_labels = all_unique_labels
        self.labels_weights = torch.from_numpy(labels_weights).to(torch.float32)

        self.template = "Next title paragraph is {}"

    def tokenize(self, sample):
        text = "\n".join(sample["input_title_sequence"])
        label = sample["immediate_next_title"]
        label_ent = self.config.label2id["entailment"]
        label_contr = self.config.label2id["contradiction"]

        if self.training:
            wrong_label = np.random.choice(self.all_unique_labels[self.all_unique_labels != label])

            encoded_sequence = self.tokenizer([(text, self.template.format(label)),
                                               (text, self.template.format(wrong_label))],

                                              truncation=True)
            encoded_sequence["labels"] = [label_ent, label_contr]
        else:
            encoded_sequence = self.tokenizer(text, self.template.format(label), truncation=True)
            encoded_sequence["labels"] = [label_ent]

        return encoded_sequence

    def prepare_input(self, batch):
        input_dict = {}

        if self.training:
            # for each batch entry, we have 2 inputs: correct label (entailment), wrong label (contradiction).
            # we need to flatten and pad accordingly
            flat_input_ids = itertools.chain.from_iterable(batch["input_ids"])
            flat_token_type_ids = itertools.chain.from_iterable(batch["token_type_ids"])
            flat_attention_mask = itertools.chain.from_iterable(batch["attention_mask"])
        else:
            flat_input_ids = batch["input_ids"]
            flat_token_type_ids = batch["token_type_ids"]
            flat_attention_mask = batch["attention_mask"]

        flat_labels = batch["labels"].flatten()  # already a tensor

        input_ids = pad_sequence(flat_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        token_type_ids = pad_sequence(flat_token_type_ids, batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(flat_attention_mask, batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)

        input_dict["input_ids"] = input_ids.to(self.device)
        input_dict["token_type_ids"] = token_type_ids.to(self.device)
        input_dict["attention_mask"] = attention_mask.to(self.device)

        if "labels" in batch:
            input_dict["labels"] = flat_labels.to(self.device)

        return input_dict

    def train_step(self, batch):
        output = self(**batch)

        entail_contradiction_logits = output.logits[:, [self.config.label2id["contradiction"], self.config.label2id["entailment"]]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1]

        loss = torch.nn.functional.binary_cross_entropy(
            prob_label_is_true,
            batch["labels"].float()
        )

        return output.logits, loss

    @torch.no_grad()
    def valid_step(self, batch):
        output = self(**batch)

        entail_contradiction_logits = output.logits[:, [self.config.label2id["contradiction"], self.config.label2id["entailment"]]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1]

        loss = torch.nn.functional.binary_cross_entropy(
            prob_label_is_true,
            batch["labels"].float()
        )

        predictions = probs.argmax(1)

        truth = batch['labels']

        match = (predictions == truth).sum()

        return match, loss
