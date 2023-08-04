import os
import random
from math import ceil
from typing import List

import numpy as np
import torch
import transformers

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5TokenizerFast, Adafactor

from tqdm import tqdm

from src import ROOT_PATH
from src.model.lm.templates import BoolNextTitlePrediction, NextTitlePrediction
from src.utils import seed_everything
from task_templates import Task

RANDOM_STATE = 42


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


def preprocess(sample, task_list):
    title_sequence = sample["input_title_sequence"]
    next_title = sample["immediate_next_title"]

    task = random.choice(task_list)

    input_text, target_text = task(title_sequence, next_title)

    encoded_sequence = tokenizer(text=input_text, text_target=target_text, truncation=True)
    encoded_sequence["immediate_next_title"] = next_title
    return encoded_sequence


if __name__ == "__main__":

    seed_everything(RANDOM_STATE)

    # PARAMETERS

    device = "cuda:0"
    max_epochs = 10
    batch_size = 4
    eval_batch_size = 2
    num_workers = 4

    tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')

    dataset = load_dataset(os.path.join(ROOT_PATH, "src", "data", "hf_dataset_script"))

    all_labels = np.unique(np.array([el
                                     for split in dataset
                                     for element in dataset[split]
                                     for el in element["title_sequence"]]))

    train = dataset["train"]
    val = dataset["validation"]
    test = dataset["test"]

    optimizer = Adafactor(
        list(model.parameters()),
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

    task_list = [NextTitlePrediction(), BoolNextTitlePrediction(all_labels)]
    model.train()

    sampled_val = val.map(sample_sequence, remove_columns=val.column_names, load_from_cache_file=False)
    preprocessed_val = sampled_val.map(lambda x: preprocess(x, task_list), remove_columns=sampled_val.column_names)
    preprocessed_val.set_format("torch")

    for epoch in range(0, max_epochs):

        # at the start of each iteration, we randomy sample the train sequence and tokenize it
        sampled_train = train.map(sample_sequence,
                                  remove_columns=train.column_names,
                                  load_from_cache_file=False,
                                  keep_in_memory=True)
        preprocessed_train = sampled_train.map(lambda x: preprocess(x, task_list),
                                               remove_columns=sampled_train.column_names,
                                               load_from_cache_file=False,
                                               keep_in_memory=True)
        preprocessed_train.set_format("torch")

        assert train.num_rows == preprocessed_train.num_rows

        pbar = tqdm(preprocessed_train.iter(batch_size=batch_size),
                    total=ceil(preprocessed_train.num_rows / batch_size))

        train_loss = 0

        model.train()

        for i, batch in enumerate(pbar):

            optimizer.zero_grad()

            input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_mask = pad_sequence(batch["attention_mask"], batch_first=True,
                                          padding_value=tokenizer.pad_token_id)
            lm_labels = pad_sequence(batch["labels"], batch_first=True, padding_value=tokenizer.pad_token_id)

            lm_labels[lm_labels == tokenizer.pad_token_id] = -100

            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=lm_labels.to(device),
            )

            loss = output.loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            if (i % 10) == 0:
                pbar.set_description(f"Epoch {epoch}, Loss -> {train_loss / (i + 1)}")

        pbar.close()

        print("VALIDATION")
        model.eval()

        pbar_val = tqdm(preprocessed_val.iter(batch_size=eval_batch_size),
                        total=ceil(preprocessed_val.num_rows / eval_batch_size))

        val_loss = 0
        matches = 0

        for i, batch in enumerate(pbar_val):

            input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_mask = pad_sequence(batch["attention_mask"], batch_first=True,
                                          padding_value=tokenizer.pad_token_id)
            lm_labels = pad_sequence(batch["labels"], batch_first=True, padding_value=tokenizer.pad_token_id)
            target_text = batch["immediate_next_title"]

            lm_labels[lm_labels == tokenizer.pad_token_id] = -100

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            lm_labels = lm_labels.to(device)

            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=lm_labels,
                )

            loss = output.loss
            val_loss += loss.item()

            num_return_sentences = 5

            beam_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                num_beams=30,
                no_repeat_ngram_size=0,
                num_return_sequences=num_return_sentences,  # top-10 recommendation
                early_stopping=True
            )
            generated_sents = tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)

            matches += sum(
                [1 if truth in generated_sents[i * num_return_sentences:(i + 1) * num_return_sentences] else 0
                 for i, truth in enumerate(target_text)])

            if (i % 10) == 0:
                pbar_val.set_description(f"Epoch {epoch}, Val Loss -> {(val_loss / (i + 1)):.4f}")

        print(matches / val.num_rows)

        pbar_val.close()

    print("EVALUATION")
    model.eval()

    test_task = [NextTitlePrediction()]
    sampled_test = test.map(sample_sequence, remove_columns=test.column_names, load_from_cache_file=False)
    preprocessed_test = sampled_test.map(lambda x: preprocess(x, test_task), remove_columns=sampled_test.column_names)
    preprocessed_test.set_format("torch")

    pbar_test = tqdm(preprocessed_test.iter(batch_size=eval_batch_size),
                     total=ceil(preprocessed_test.num_rows / eval_batch_size))

    matches = 0
    for i, batch in enumerate(pbar_test):

        input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(batch["attention_mask"], batch_first=True,
                                      padding_value=tokenizer.pad_token_id)
        target_text = batch["immediate_next_title"]

        num_return_sentences = 5

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        beam_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            num_beams=30,
            no_repeat_ngram_size=0,
            num_return_sequences=num_return_sentences,  # top-10 recommendation
            early_stopping=True
        )
        generated_sents = tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)

        # match with actual labels
        embeddings1 = sim_model.encode(generated_sents, convert_to_tensor=True)
        embeddings2 = sim_model.encode(all_labels, convert_to_tensor=True)

        sim = util.cos_sim(embeddings1, embeddings2)
        mapped_predictions = [all_labels[index] for index in sim.argmax(axis=1)]

        matches += sum(
            [1 if truth in mapped_predictions[i * num_return_sentences:(i + 1) * num_return_sentences] else 0
             for i, truth in enumerate(target_text)])

        if (i % 10) == 0 and i != 0:
            pbar_test.set_description(f"Acc -> {(matches / i):.4f}")

    print(matches / test.num_rows)
