import random

import numpy as np
import torch
from sentence_transformers import util
from torch.nn.utils.rnn import pad_sequence
from transformers import T5ForConditionalGeneration, Adafactor

from src.data.clustering import ClusterLabelMapper
from src.model.lm.t5.templates import NextTitlePrediction, BoolNextTitlePrediction
from src.model.sequence_classification.seq_models_interface import SeqClassification
from src.sentence_encoders import SentenceEncoder


class FineTunedFlanT5(T5ForConditionalGeneration, SeqClassification):

    def __init__(self, config, sentence_encoder: SentenceEncoder, all_labels: np.ndarray, tokenizer, device,
                 num_return_sequences=5, max_new_tokens=50, num_beams=30, no_repeat_ngram_size=0, early_stopping=True):

        T5ForConditionalGeneration.__init__(self, config)

        optimizer = Adafactor(
            list(self.parameters()),
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

        SeqClassification.__init__(
            self,
            tokenizer=tokenizer,
            optimizer=optimizer
        )

        self.num_return_sequences = num_return_sequences
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.early_stopping = early_stopping

        self.all_labels = all_labels
        self.sim_model = sentence_encoder
        self.encoded_all_labels = self.sim_model(*all_labels, desc="Encoding ALL labels for FlanT5...")
        self.task_list = [NextTitlePrediction(), BoolNextTitlePrediction(self.all_labels)]
        self.test_task = NextTitlePrediction()  # for testing we want that the model predicts the title as output

        self.to(device)

    def tokenize(self, sample, fit_label_cluster_mapper: ClusterLabelMapper = None):

        title_sequence = sample["input_title_sequence"]
        next_title = sample["immediate_next_title"]

        task = random.choice(self.task_list) if self.training else self.test_task

        input_text, target_text = task(title_sequence, next_title)

        if fit_label_cluster_mapper is not None:
            immediate_next_cluster = fit_label_cluster_mapper.get_clusters_from_labels(sample["immediate_next_title"])
            additional_info = f"\nNext title cluster: {immediate_next_cluster}"
            input_text = input_text + additional_info

        encoded_sequence = self.tokenizer(text=input_text, text_target=target_text, truncation=True)
        encoded_sequence["immediate_next_title"] = next_title
        return encoded_sequence

    def prepare_input(self, batch):
        input_dict = {}

        input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(batch["attention_mask"], batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)

        input_dict["input_ids"] = input_ids.to(self.device)
        input_dict["attention_mask"] = attention_mask.to(self.device)

        if "labels" in batch:
            lm_labels = pad_sequence(batch["labels"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100

            input_dict["labels"] = lm_labels.to(self.device)

        if not self.training:
            input_dict["immediate_next_title"] = batch["immediate_next_title"]

        return input_dict

    def train_step(self, batch):

        output = self(**batch)

        return output.logits, output.loss

    @torch.no_grad()
    def valid_step(self, batch):

        target_text = batch.pop("immediate_next_title")
        output = self(**batch)

        beam_outputs = self.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            num_return_sequences=self.num_return_sequences,
            early_stopping=self.early_stopping,
        )

        generated_sents = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        encoded_preds = self.sim_model.encode_batch(generated_sents)

        sim = util.cos_sim(encoded_preds, self.encoded_all_labels).numpy()
        mapped_predictions = self.all_labels[sim.argmax(axis=1)]

        val_loss = output.loss

        matches = sum(
            [truth in mapped_predictions[j * self.num_return_sequences:(j + 1) * self.num_return_sequences]
             for j, truth in enumerate(target_text)])

        return matches, val_loss
