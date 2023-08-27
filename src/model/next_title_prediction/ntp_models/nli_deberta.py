import itertools
import random
from math import ceil

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DebertaV2ForSequenceClassification

from src.model.clustering import ClusterLabelMapper
from src.model.next_title_prediction.ntp_models_interface import NextTitlePredictor


class NextTitleNliDeberta(NextTitlePredictor):

    model_class = DebertaV2ForSequenceClassification

    def __init__(self,
                 model: DebertaV2ForSequenceClassification,
                 all_unique_labels: np.ndarray,
                 tokenizer,
                 validation_mini_batch_size: int = 16,
                 cluster_label_mapper: ClusterLabelMapper = None,
                 device: str = 'cuda:0'):

        NextTitlePredictor.__init__(
            self,
            model=model,
            tokenizer=tokenizer,
            optimizer=torch.optim.AdamW(list(model.parameters()), lr=2e-5),
            cluster_label_mapper=cluster_label_mapper,
            device=device
        )

        self.all_unique_labels = all_unique_labels

        self.template = "Next title paragraph is {}"
        self.validation_mini_batch_size = validation_mini_batch_size

    def tokenize(self, sample):

        if self.cluster_label_mapper is not None:
            immediate_next_cluster = self.cluster_label_mapper.get_clusters_from_labels(sample["immediate_next_title"])
            next_candidate_titles = self.cluster_label_mapper.get_labels_from_clusters(immediate_next_cluster)
            text = ", ".join(sample["input_title_sequence"]) + f"\nNext title cluster: {immediate_next_cluster}"
        else:
            next_candidate_titles = self.all_unique_labels
            text = ", ".join(sample["input_title_sequence"])

        label = sample["immediate_next_title"]
        label_ent: int = self.config.label2id["entailment"]
        label_contr: int = self.config.label2id["contradiction"]

        if self.training:

            wrong_label = random.choice(next_candidate_titles[next_candidate_titles != label])
            encoded_sequence = self.tokenizer([(text, self.template.format(label)),
                                               (text, self.template.format(wrong_label))],
                                              truncation=True)

            encoded_sequence["labels"] = [label_ent, label_contr]

        else:

            encoded_sequence = self.tokenizer([(text, self.template.format(candidate_title))
                                               for candidate_title in next_candidate_titles],
                                              return_tensors='pt',
                                              padding=True,
                                              truncation=True)

            encoded_sequence["labels"] = torch.full(size=next_candidate_titles.shape, fill_value=label_contr)

            next_possible_titles_correct_idx = next_candidate_titles == label
            encoded_sequence["labels"][next_possible_titles_correct_idx] = label_ent

            encoded_sequence["labels"] = encoded_sequence["labels"].tolist()

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
        label_contr = self.config.label2id["contradiction"]
        label_ent = self.config.label2id["entailment"]

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

            # get index of label which is entailment in truth tensor
            truth_index = (truth == self.config.label2id["entailment"]).nonzero().item()

            # increment accuracy if they match
            acc += 1 if prediction_index == truth_index else 0

            val_loss += loss

        return acc, val_loss
