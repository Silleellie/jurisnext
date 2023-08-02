import itertools
import os
import random

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_metric
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, TrainingArguments, \
    BartForSequenceClassification, Trainer, BartTokenizerFast

from src import ROOT_PATH, DATA_DIR


def sample_sequence(batch):
    assert len(batch["text_sequence"]) == len(batch["title_sequence"])

    # a sequence has at least 1 data point, but it can have more depending on the length of the sequence
    # We must ensure that at least an element can be used as test set
    # in the "sliding_training_size" is included the target item
    sliding_size = random.randint(1, len(batch["text_sequence"]) - 1)

    # TO DO: consider starting always from the initial paragraph,
    # rather than varying the starting point of the seq
    start_index = random.randint(0, len(batch["text_sequence"]) - sliding_size - 1)
    end_index = start_index + sliding_size

    return {
        "case_id": batch["case_id"],
        "input_text_sequence": batch["text_sequence"][start_index:end_index],
        "input_title_sequence": batch["title_sequence"][start_index:end_index],
        "input_keywords_sequence": batch["rel_keywords_sequence"][start_index:end_index],
        "immediate_next_text": batch["text_sequence"][end_index],
        "immediate_next_title": batch["title_sequence"][end_index],
        "immediate_next_rel_keywords": batch["rel_keywords_sequence"][end_index]
    }


def preprocess(sample):
    text = "\n".join(sample["input_title_sequence"])
    label = sample["immediate_next_title"]
    label_int = model.config.label2id["entailment"]

    if not random.getrandbits(1):
        label = np.random.choice(all_labels[all_labels != label])
        label_int = model.config.label2id["contradiction"]

    encoded_sequence = tokenizer(text, template.format(label), truncation=True)
    encoded_sequence["labels"] = label_int
    encoded_sequence["input_sentence"] = tokenizer.batch_decode(encoded_sequence.input_ids)
    return encoded_sequence


def preprocess_test(sample):
    text = "\n".join(sample["input_title_sequence"])
    label = sample["immediate_next_title"]

    wrong_labels = list(all_labels[all_labels != label])
    random.shuffle(wrong_labels)

    wrong_labels_cut = wrong_labels[:50]
    candidate_labels = wrong_labels_cut + [label]

    candidate_targets = [template.format(cand_label) for cand_label in candidate_labels]

    encoded_sequence = tokenizer(list(zip(itertools.repeat(text), candidate_targets)), truncation=True)
    return encoded_sequence


def compute_metrics(p: EvalPrediction):
    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"]
    result["f1"] = metric_f1.compute(predictions=preds, references=p.label_ids, average='macro')["f1"]
    return result


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-xsmall")
    model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-xsmall")

    dataset = load_dataset(os.path.join(ROOT_PATH, "src", "data", "hf_dataset_script"))

    all_labels = np.unique(np.array([el
                                     for split in dataset
                                     for element in dataset[split]
                                     for el in element["title_sequence"]]))

    template = "Next title paragraph is {}"

    train = dataset["train"]
    val = dataset["validation"]
    test = dataset["test"]

    sampled_train = train.map(sample_sequence, remove_columns=train.column_names, load_from_cache_file=False)
    preprocessed_train = sampled_train.map(preprocess, remove_columns=sampled_train.column_names)

    sampled_val = val.map(sample_sequence, remove_columns=val.column_names, load_from_cache_file=False)
    preprocessed_val = sampled_val.map(preprocess, remove_columns=sampled_val.column_names)

    sampled_test: datasets.Dataset = test.map(sample_sequence, remove_columns=test.column_names, load_from_cache_file=False)
    preprocessed_test = sampled_test.map(preprocess_test)
    preprocessed_test = preprocessed_test.select_columns(["input_ids", "token_type_ids",
                                                          "attention_mask", "immediate_next_title"])
    preprocessed_test.set_format("torch")

    # training_args = TrainingArguments(
    #     output_dir="we",  # Output directory
    #     num_train_epochs=1,  # Total number of training epochs
    #     per_device_train_batch_size=4,  # Batch size per device during training
    #     per_device_eval_batch_size=4,  # Batch size for evaluation
    #     warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    #     weight_decay=0.01,  # Strength of weight decay
    #     logging_steps=10
    # )
    #
    # trainer = Trainer(
    #     model=model,  # The instantiated model to be trained
    #     args=training_args,  # Training arguments, defined above
    #     compute_metrics=compute_metrics,  # A function to compute the metrics
    #     train_dataset=preprocessed_train,  # Training dataset
    #     eval_dataset=preprocessed_val,
    #     tokenizer=tokenizer,  # The tokenizer that was used
    # )
    #
    # trainer.train()
    #
    # print(trainer.evaluate())

    # model is now the fine tuned one, no need to reload it
    model = AutoModelForSequenceClassification.from_pretrained("we/checkpoint-500").to("cuda:0")

    eval_batch_size = 4

    it = preprocessed_test.iter(batch_size=eval_batch_size)

    model.eval()
    count_ok = 0
    den = 0
    for sample in tqdm(it):

        input_ids = pad_sequence(sample["input_ids"], batch_first=True, padding_value=tokenizer.pad_token_id)
        token_type_ids = pad_sequence(sample["token_type_ids"], batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(sample["attention_mask"], batch_first=True, padding_value=tokenizer.pad_token_id)

        input_ids.to("cuda:0")
        token_type_ids.to("cuda:0")
        attention_mask.to("cuda:0")

        with torch.no_grad():
            scores = model(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask).logits

        label_mapping = ['contradiction', 'entailment', 'neutral']
        labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]

        if labels[-1] == "entailment":
            count_ok += 1

        den += 1

    acc = count_ok / den

    print(acc)
