import os
from math import ceil

import datasets
import numpy as np
import torch
from datasets import load_dataset

from sentence_transformers import util, SentenceTransformer

from tqdm import tqdm

from src import RANDOM_STATE, ROOT_PATH
from src.data.dataset_map_fn import sample_sequence
from src.model.lm.t5.flan_t5 import FineTunedFlanT5
from src.utils import seed_everything


class LMTrainer:

    def __init__(self, n_epochs, batch_size,
                 model: [FineTunedFlanT5],
                 all_labels: np.ndarray,
                 eval_batch_size=None, num_workers=4,
                 num_return_sequences=5, max_new_tokens=50, num_beams=30, no_repeat_ngram_size=0, early_stopping=True):

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = model
        self.all_labels = all_labels
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.num_return_sequences = num_return_sequences
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.early_stopping = early_stopping
        self.device = model.device

        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        self.encoded_labels = self.sim_model.encode(all_labels, convert_to_tensor=True)

    def train(self, train_dataset: datasets.Dataset, validation_dataset: datasets.Dataset = None):

        # validation set remains the same and should NOT be sampled at each epoch, otherwise
        # results are not comparable
        model.eval()  # eval because the "tokenize" function select different tasks depending on the mode
        sampled_val = validation_dataset.map(sample_sequence,
                                             remove_columns=validation_dataset.column_names,
                                             load_from_cache_file=False)
        preprocessed_val = sampled_val.map(self.model.tokenize, remove_columns=sampled_val.column_names)
        preprocessed_val.set_format("torch")
        model.train()

        # ceil because we don't drop the last batch
        total_n_batch = ceil(train_dataset.num_rows / self.batch_size)

        for epoch in range(0, self.n_epochs):

            # at the start of each iteration, we randomly sample the train sequence and tokenize it
            sampled_train = train_dataset.map(sample_sequence,
                                              remove_columns=train_dataset.column_names,
                                              load_from_cache_file=False,
                                              keep_in_memory=True)
            preprocessed_train = sampled_train.map(self.model.tokenize,
                                                   remove_columns=sampled_train.column_names,
                                                   load_from_cache_file=False,
                                                   keep_in_memory=True)
            preprocessed_train.set_format("torch")

            pbar = tqdm(preprocessed_train.iter(batch_size=self.batch_size),
                        total=total_n_batch)

            train_loss = 0
            self.model.train()
            for i, batch in enumerate(pbar, start=1):

                self.model.optimizer.zero_grad()

                prepared_input = self.model.prepare_input(batch)
                output = self.model(**prepared_input)

                loss = output.loss
                train_loss += loss.item()

                loss.backward()
                self.model.optimizer.step()

                # we update the loss every 1% progress considering the total n° of batches
                if (i % (total_n_batch // 100)) == 0:
                    pbar.set_description(f"Epoch {epoch}, Loss -> {train_loss / (i + 1)}")

            pbar.close()

            if validation_dataset is not None:
                self.validation(preprocessed_validation=preprocessed_val)

    def validation(self, preprocessed_validation: datasets.Dataset):
        print("VALIDATION")
        self.model.eval()

        # ceil because we don't drop the last batch
        total_n_batch = ceil(preprocessed_validation.num_rows / self.eval_batch_size)

        pbar_val = tqdm(preprocessed_validation.iter(batch_size=self.eval_batch_size),
                        total=total_n_batch)

        val_loss = 0
        matches = 0

        for i, batch in enumerate(pbar_val, start=1):

            prepared_input = self.model.prepare_input(batch)
            target_text = batch["immediate_next_title"]

            with torch.no_grad():
                output = self.model(**prepared_input)

                beam_outputs = self.model.generate(
                    input_ids=prepared_input["input_ids"],
                    attention_mask=prepared_input["attention_mask"],
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    num_return_sequences=self.num_return_sequences,
                    early_stopping=self.early_stopping,
                )
                generated_sents = self.model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)

            # match with actual labels by taking the most similiar to the label generated
            # by the language model
            encoded_preds = self.sim_model.encode(generated_sents, convert_to_tensor=True)

            sim = util.cos_sim(encoded_preds, self.encoded_labels)
            mapped_predictions = self.all_labels[sim.cpu().argmax(axis=1)]

            val_loss += output.loss

            matches += sum(
                [truth in mapped_predictions[j * self.num_return_sequences:(j + 1) * self.num_return_sequences]
                 for j, truth in enumerate(target_text)])

            # we update the loss every 1% progress considering the total n° of batches
            if (i % (total_n_batch // 100)) == 0:
                pbar_val.set_description(f"Val Loss -> {(matches / (i * len(batch))):.4f}")

        print(matches / preprocessed_validation.num_rows)

        pbar_val.close()

    def evaluate(self, test_dataset: datasets.Dataset):

        print("EVALUATION")
        self.model.eval()
        sampled_test = test_dataset.map(sample_sequence,
                                        remove_columns=test_dataset.column_names, load_from_cache_file=False)
        preprocessed_test = sampled_test.map(self.model.tokenize,
                                             remove_columns=sampled_test.column_names)
        preprocessed_test.set_format("torch")

        total_n_batch = ceil(preprocessed_test.num_rows / self.eval_batch_size)

        pbar_test = tqdm(preprocessed_test.iter(batch_size=self.eval_batch_size),
                         total=total_n_batch)

        matches = 0
        for i, batch in enumerate(pbar_test, start=1):

            prepared_input = self.model.prepare_input(batch)
            target_text = batch["immediate_next_title"]

            beam_outputs = self.model.generate(
                input_ids=prepared_input["input_ids"],
                attention_mask=prepared_input["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                num_return_sequences=self.num_return_sequences,
                early_stopping=self.early_stopping
            )
            generated_sents = self.model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)

            # match with actual labels by taking the most similiar to the label generated
            # by the language model
            encoded_preds = self.sim_model.encode(generated_sents, convert_to_tensor=True)

            sim = util.cos_sim(encoded_preds, self.encoded_labels)
            mapped_predictions = self.all_labels[sim.argmax(axis=1)]

            matches += sum(
                [truth in mapped_predictions[i * self.num_return_sequences:(j + 1) * self.num_return_sequences]
                 for j, truth in enumerate(target_text)])

            # we update the loss every 1% progress considering the total n° of batches
            if (i % (total_n_batch // 100)) == 0:
                pbar_test.set_description(f"Acc -> {(matches / (i * len(batch))):.4f}")

        print(matches / preprocessed_test.num_rows)


if __name__ == "__main__":
    seed_everything(RANDOM_STATE)

    # PARAMETERS
    dataset = load_dataset(os.path.join(ROOT_PATH, "src", "data", "hf_dataset_script"))

    all_labels = np.unique(np.array([el
                                     for split in dataset
                                     for element in dataset[split]
                                     for el in element["title_sequence"]]))

    train = dataset["train"]
    val = dataset["validation"]
    test = dataset["test"]

    model = FineTunedFlanT5.from_pretrained("google/flan-t5-small", all_labels=all_labels).to("cuda:0")

    trainer = LMTrainer(
        n_epochs=1,
        batch_size=4,
        model=model,
        all_labels=all_labels,
        eval_batch_size=2
    )

    trainer.train(train, val)

    trainer.evaluate(test)
