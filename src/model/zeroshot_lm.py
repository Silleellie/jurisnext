import os
import random

import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, TrainingArguments, \
    BartForSequenceClassification, Trainer, BartTokenizerFast

from src import ROOT_PATH, DATA_DIR

if __name__ == "__main__":

    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-mnli')

    dataset = load_dataset(os.path.join(ROOT_PATH, "src", "data", "hf_dataset_script"))

    all_labels = np.unique(np.array([el
                                     for split in dataset
                                     for element in dataset[split]
                                     for el in element["title_sequence"]]))

    template = "Next title paragraph is {}"

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
        label_int = 2

        if not random.getrandbits(1):
            label = np.random.choice(all_labels[all_labels != label])
            label_int = 0

        encoded_sequence = tokenizer(text, template.format(label), truncation=True)
        encoded_sequence["labels"] = label_int
        encoded_sequence["input_sentence"] = tokenizer.batch_decode(encoded_sequence.input_ids)
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

    train = dataset["train"]
    val = dataset["validation"]

    sampled_train = train.map(sample_sequence, remove_columns=train.column_names, load_from_cache_file=False)
    preprocessed_train = sampled_train.map(preprocess, remove_columns=sampled_train.column_names)

    sampled_val = val.map(sample_sequence, remove_columns=val.column_names, load_from_cache_file=False)
    preprocessed_val = sampled_val.map(preprocess, remove_columns=sampled_val.column_names)

    training_args = TrainingArguments(
        output_dir="we",  # Output directory
        num_train_epochs=1,  # Total number of training epochs
        per_device_train_batch_size=1,  # Batch size per device during training
        per_device_eval_batch_size=1,  # Batch size for evaluation
        warmup_steps=500,  # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # Strength of weight decay
        logging_steps=10
    )

    model = BartForSequenceClassification.from_pretrained("facebook/bart-large-mnli",
                                                          num_labels=len(all_labels),
                                                          ignore_mismatched_sizes=True)

    trainer = Trainer(
        model=model,  # The instantiated model to be trained
        args=training_args,  # Training arguments, defined above
        compute_metrics=compute_metrics,  # A function to compute the metrics
        train_dataset=preprocessed_train,  # Training dataset
        eval_dataset=preprocessed_val,
        tokenizer=tokenizer,  # The tokenizer that was used
        device=0
    )

    trainer.train()
