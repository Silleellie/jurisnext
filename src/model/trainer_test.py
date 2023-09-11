import copy
import random

import datasets
import evaluate
import torch
from cytoolz.dicttoolz import merge_with
import numpy as np
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, BertForSequenceClassification, BertTokenizerFast, \
    DataCollatorWithPadding, BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer

from src.data.legal_dataset import LegalDataset
from src.utils import seed_everything

accuracy = evaluate.load("accuracy")


def preprocess(sample):
    return_dict = tokenizer(', '.join(sample["input_title_sequence"]), truncation=True)
    return_dict["label"] = model.config.label2id[sample["immediate_next_title"]]

    return return_dict


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)


# class CustomTrainer(Trainer):
#
#     def get_train_dataloader(self) -> DataLoader:
#         """
#         Returns the training [`~torch.utils.data.DataLoader`].
#
#         Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
#         training if necessary) otherwise.
#
#         Subclass and override this method if you want to inject some custom behavior.
#         """
#
#         bak_ds = copy.deepcopy(self.train_dataset)
#
#         train_ds: datasets.Dataset = self.train_dataset
#         # at the start of each iteration, we randomly sample the train sequence and tokenize it
#         sampled_train = train_ds.map(LegalDataset.perform_sampling,
#                                      remove_columns=train_dataset.column_names,
#                                      load_from_cache_file=False,
#                                      keep_in_memory=True)
#         preprocessed_train = sampled_train.map(preprocess,
#                                                remove_columns=sampled_train.column_names,
#                                                load_from_cache_file=False,
#                                                keep_in_memory=True)
#         preprocessed_train.set_format("torch")
#
#         self.train_dataset = preprocessed_train
#
#         train_dl = super().get_train_dataloader()
#
#         self.train_dataset = bak_ds
#
#         return train_dl


if __name__ == "__main__":
    seed_everything(42)

    ds = LegalDataset.load_dataset()
    dataset = ds.get_hf_datasets()
    all_unique_labels = ds.all_unique_labels

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(all_unique_labels),
        problem_type="single_label_classification",
        label2id={x: i for i, x in enumerate(all_unique_labels)},
        id2label={i: x for i, x in enumerate(all_unique_labels)},
        ignore_mismatched_sizes=True
    )

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    tokenizer.add_tokens(list(ds.all_ner_tokens))
    model.resize_token_embeddings(len(tokenizer))

    preprocessed_train = train_dataset.map(preprocess,
                                           remove_columns=train_dataset.column_names,
                                           load_from_cache_file=False)
    preprocessed_train.set_format("torch")

    preprocessed_val = val_dataset.map(preprocess,
                                       remove_columns=val_dataset.column_names,
                                       load_from_cache_file=False)
    preprocessed_val.set_format("torch")

    training_args = TrainingArguments(
        "test-trainer",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        logging_steps=0.01,
        report_to="none",
        save_strategy="no",
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=preprocessed_train,
        eval_dataset=preprocessed_val,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
