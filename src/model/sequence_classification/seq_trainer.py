import os
from math import ceil

import datasets
import numpy as np
import torch
from datasets import load_dataset
from sklearn.utils import compute_class_weight

from tqdm import tqdm

from src import RANDOM_STATE, ROOT_PATH
from src.data.dataset_map_fn import sample_sequence
from src.model.sequence_classification.bert.bert import FineTunedBert
from src.utils import seed_everything


class SeqTrainer:

    def __init__(self,
                 n_epochs, batch_size,
                 model,
                 all_labels: np.ndarray,
                 device='cuda:0',
                 eval_batch_size=None,
                 num_workers=4,
                 output_name=None):

        self.model = model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.all_labels = all_labels
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.device = device

        # output name
        if output_name is None:
            output_name = f"{model.config.name_or_path}_{n_epochs}" if output_name is None else output_name

        self.output_path = os.path.join(MODELS_DIR, output_name)

    def train(self, train_dataset: datasets.Dataset, validation_dataset: datasets.Dataset = None):

        # validation set remains the same and should NOT be sampled at each epoch, otherwise
        # results are not comparable
        self.model.eval()  # eval because the "tokenize" function select different tasks depending on the mode
        sampled_val = validation_dataset.map(sample_sequence,
                                             remove_columns=validation_dataset.column_names,
                                             load_from_cache_file=False)
        preprocessed_val = sampled_val.map(self.model.tokenize, remove_columns=sampled_val.column_names)
        preprocessed_val.set_format("torch")
        self.model.train()

        # ceil because we don't drop the last batch
        total_n_batch = ceil(train_dataset.num_rows / self.batch_size)

        # early stopping parameters
        min_val_loss = np.inf
        no_change_counter = 0
        min_delta = 1e-4

        for epoch in range(0, self.n_epochs):

            # if no significant change happens to the loss after 10 epochs then early stopping
            if no_change_counter == 10:
                print("Early stopping")
                self.model.save_pretrained(self.output_path)
                break

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
                output, loss = self.model.train_step(prepared_input)

                loss.backward()
                self.model.optimizer.step()

                train_loss += loss.item()

                # we update the loss every 1% progress considering the total n° of batches
                if (i % ceil(total_n_batch / 100)) == 0:
                    pbar.set_description(f"Epoch {epoch}, Loss -> {(train_loss / i):.6f}")

            pbar.close()

            if validation_dataset is not None:
                val_loss = self.validation(preprocessed_validation=preprocessed_val)

                # if there is a significant difference between the last minimum loss and the current one
                # set it as the new min loss and save the model parameters
                if (min_val_loss - val_loss) > min_delta:

                    min_val_loss = val_loss
                    no_change_counter = 0
                    self.model.save_pretrained(self.output_path)

                else:

                    no_change_counter += 1

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
            acc, loss = self.model.valid_step(prepared_input)

            val_loss += loss.item()
            matches += acc

            # we update the loss every 1% progress considering the total n° of batches
            if (i % ceil(total_n_batch / 100)) == 0:
                pbar_val.set_description(f"Val Loss -> {(val_loss / i):.6f}")

        print(matches / preprocessed_validation.num_rows)

        pbar_val.close()

        return val_loss / len(pbar_val)

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
            acc, _ = self.model.valid_step(prepared_input)

            matches += acc

            # we update the loss every 1% progress considering the total n° of batches
            if (i % ceil(total_n_batch / 100)) == 0:
                pbar_test.set_description(f"Acc -> {(matches / (i * self.eval_batch_size)):.6f}")

        print(matches / preprocessed_test.num_rows)


if __name__ == "__main__":
    seed_everything(RANDOM_STATE)

    # PARAMETERS
    n_epochs = 100
    batch_size = 2
    eval_batch_size = 2

    dataset = load_dataset(os.path.join(ROOT_PATH, "src", "data", "hf_dataset_script"))

    all_labels_occurrences = np.array([el
                                       for split in dataset
                                       for element in dataset[split]
                                       for el in element["title_sequence"]])

    all_unique_labels = np.unique(all_labels_occurrences)

    labels_weights = compute_class_weight(class_weight='balanced', classes=all_unique_labels, y=all_labels_occurrences)

    model = FineTunedBert.from_pretrained('bert-base-uncased',
                                          problem_type="single_label_classification",
                                          num_labels=len(all_unique_labels),
                                          id2label={idx: label for idx, label in
                                                    enumerate(all_unique_labels)},
                                          label2id={label: idx for idx, label in
                                                    enumerate(all_unique_labels)},
                                          labels_weights=labels_weights
                                          ).to('cuda:0')

    train = dataset["train"]
    val = dataset["validation"]
    test = dataset["test"]

    trainer = SeqTrainer(
        model=model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        all_labels=all_unique_labels,
        eval_batch_size=eval_batch_size
    )

    trainer.train(train, val)

    trainer.model = FineTunedNliDeberta.from_pretrained(trainer.output_path,

                                                        all_unique_labels=all_unique_labels,
                                                        labels_weights=labels_weights,
                                                        tokenizer=tokenizer)

    trainer.evaluate(test)
