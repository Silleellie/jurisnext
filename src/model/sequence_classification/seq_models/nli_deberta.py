import itertools
import random
from math import ceil

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DebertaV2ForSequenceClassification

from src.model.sequence_classification.seq_models_interface import SeqClassification


class FineTunedNliDeberta(DebertaV2ForSequenceClassification, SeqClassification):

    def __init__(self, config, all_unique_labels: np.ndarray, tokenizer,
                 title_to_cluster_title: dict, cluster_title_to_possible_titles: dict,
                 validation_mini_batch_size: int = 16):

        DebertaV2ForSequenceClassification.__init__(self, config)

        SeqClassification.__init__(
            self,
            tokenizer=tokenizer,
            optimizer=torch.optim.AdamW(list(self.parameters()), lr=2e-5)
        )

        self.all_unique_labels = all_unique_labels

        self.template = "Next title paragraph is {}"

        self.title_to_cluster_title = title_to_cluster_title
        self.cluster_title_to_possible_titles = cluster_title_to_possible_titles
        self.validation_mini_batch_size = validation_mini_batch_size

    def tokenize(self, sample):

        next_cluster_title = self.title_to_cluster_title[sample["immediate_next_title"]]
        next_possible_titles = self.cluster_title_to_possible_titles[next_cluster_title]
        next_possible_titles = np.array(next_possible_titles)
        next_possible_titles_correct_idx = np.where(next_possible_titles == sample["immediate_next_title"])[0].item()

        text = ", ".join(sample["input_title_sequence"]) + " -> " + next_cluster_title
        label = sample["immediate_next_title"]
        label_ent = self.config.label2id["entailment"]
        label_contr = self.config.label2id["contradiction"]

        if self.training:

            wrong_label = random.choice(next_possible_titles[next_possible_titles != sample["immediate_next_title"]])
            encoded_sequence = self.tokenizer([(text, self.template.format(label)),
                                               (text, self.template.format(wrong_label))],
                                              truncation=True)

            encoded_sequence["labels"] = [label_ent, label_contr]

        else:

            encoded_sequence = self.tokenizer([(text, self.template.format(sent)) for sent in next_possible_titles],
                                              return_tensors='pt',
                                              padding=True,
                                              truncation=True)

            encoded_sequence["labels"] = [label_ent if i == next_possible_titles_correct_idx else label_contr
                                          for i in range(len(next_possible_titles))]

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

            input_dict["input_ids"] = input_ids.to(self.device)
            input_dict["token_type_ids"] = token_type_ids.to(self.device)
            input_dict["attention_mask"] = attention_mask.to(self.device)

        else:

            input_dict["input_ids"] = [x.to(self.device) for x in batch["input_ids"]]
            input_dict["token_type_ids"] = [x.to(self.device) for x in batch["token_type_ids"]]
            input_dict["attention_mask"] = [x.to(self.device) for x in batch["attention_mask"]]

        if "labels" in batch:

            if self.training:
                flat_labels = batch["labels"].flatten()  # already a tensor
                input_dict["labels"] = flat_labels.to(self.device)
            else:
                input_dict["labels"] = [x.to(self.device).long() for x in batch["labels"]]

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

        return output.logits, loss

    @torch.no_grad()
    def valid_step(self, batch):

        mini_batch_size = self.validation_mini_batch_size

        val_loss = 0
        acc = 0

        for i, truth in enumerate(batch['labels']):

            output_logits = []
            # ceil to not drop the last batch
            max_j = ceil(batch['input_ids'][i].shape[0] / mini_batch_size)

            for j in range(max_j):
                mini_batch = {k: v[i][j * mini_batch_size:(j + 1) * mini_batch_size] for k, v in batch.items()}

                output = self(**mini_batch)
                output_logits.append(output.logits)

            output_logits = torch.vstack(output_logits)
            entail_contradiction_logits = output_logits[:,
                                          [self.config.label2id["contradiction"], self.config.label2id["entailment"]]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:, 1]
            truth = batch["labels"][i]

            loss = torch.nn.functional.binary_cross_entropy(
                prob_label_is_true,
                truth.float()
            )

            prediction = prob_label_is_true.argmax(0)
            truth_prediction = truth.argmax(0)

            val_loss += loss
            acc += 1 if prediction == truth_prediction else 0

        return acc, val_loss
