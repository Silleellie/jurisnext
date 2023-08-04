import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast, T5ForConditionalGeneration, Adafactor

from src.model.lm.t5.templates import NextTitlePrediction, BoolNextTitlePrediction


class FineTunedFlanT5(T5ForConditionalGeneration):

    def __init__(self, config, all_labels: np.ndarray):
        super().__init__(config)

        self.tokenizer = T5TokenizerFast.from_pretrained(config.name_or_path)

        self.optimizer = Adafactor(
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

        self.all_labels = all_labels
        self.task_list = [NextTitlePrediction(), BoolNextTitlePrediction(self.all_labels)]
        self.test_task = NextTitlePrediction()  # for testing we want that the model predicts the title as output

    def tokenize(self, sample):
        title_sequence = sample["input_title_sequence"]
        next_title = sample["immediate_next_title"]

        task = random.choice(self.task_list) if self.training else self.test_task

        input_text, target_text = task(title_sequence, next_title)

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

        return input_dict

    def train_step(self, input_ids, attention_mask, lm_labels):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
        )

        return output

    @torch.no_grad()
    def valid_step(self, input_ids, attention_mask, labels,
                   n_return_sequences=5,
                   max_new_tokens=50, num_beams=30, no_repeat_ngram_size=0,
                   early_stopping=True, **kwargs):

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        val_loss = output.loss

        generated_sents = self.generate_step(input_ids=input_ids, attention_mask=attention_mask,
                                             n_return_sequences=n_return_sequences, max_new_tokens=max_new_tokens,
                                             num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size,
                                             early_stopping=early_stopping, **kwargs)

        return val_loss, generated_sents

    @torch.no_grad()
    def generate_step(self, input_ids, attention_mask,
                      n_return_sequences=5,
                      max_new_tokens=50, num_beams=30, no_repeat_ngram_size=0,
                      early_stopping=True, **kwargs):

        beam_outputs = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=n_return_sequences,
            early_stopping=early_stopping,
            **kwargs
        )
        generated_sents = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)

        return generated_sents
