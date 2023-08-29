import os
from math import ceil
import itertools

from typing import Optional

import datasets
import numpy as np

from tqdm import tqdm

from src import MODELS_DIR
from src.data.legal_dataset import LegalDataset
from src.model.next_title_prediction.ntp_models_abtract import NTPModel
from src.utils import seed_everything


class NTPTrainer:

    def __init__(self,
                 n_epochs: int,
                 batch_size: int,
                 ntp_model: NTPModel,
                 all_labels: np.ndarray,
                 device: str = 'cuda:0',
                 eval_batch_size: Optional[int] = None,
                 output_name: Optional[str] = None):

        self.ntp_model = ntp_model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.all_labels = all_labels
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.device = device

        # output name
        if output_name is None:
            output_name = f"{ntp_model.config.name_or_path}_{n_epochs}" if output_name is None else output_name

        self.output_path = os.path.join(MODELS_DIR, output_name)

    def train(self, train_dataset: datasets.Dataset, validation_dataset: datasets.Dataset = None):

        if self.ntp_model.cluster_label_mapper is not None:
            # retrieve all unique labels which appear in train set
            flat_train_labels = itertools.chain.from_iterable(train_dataset["title_sequence"])
            unique_train_labels = np.unique(np.fromiter(flat_train_labels, dtype=object)).astype(str)
            self.ntp_model._train_clusters(unique_train_labels, self.all_labels)

        # validation set remains the same and should NOT be sampled at each epoch, otherwise
        # results are not comparable
        self.ntp_model.eval()  # eval because the "tokenize" function select different tasks depending on the mode
        preprocessed_val = validation_dataset.map(self.ntp_model.tokenize,
                                                  remove_columns=validation_dataset.column_names,
                                                  load_from_cache_file=False)
        preprocessed_val.set_format("torch")

        # ceil because we don't drop the last batch
        total_n_batch = ceil(train_dataset.num_rows / self.batch_size)

        # early stopping parameters
        min_val_loss = np.inf
        no_change_counter = 0
        min_delta = 1e-4

        for epoch in range(0, self.n_epochs):

            self.ntp_model.train()

            # if no significant change happens to the loss after 10 epochs then early stopping
            if no_change_counter == 10:
                print("Early stopping")
                self.ntp_model.save(self.output_path)
                break

            # at the start of each iteration, we randomly sample the train sequence and tokenize it
            sampled_train = train_dataset.map(LegalDataset.perform_sampling,
                                              remove_columns=train_dataset.column_names,
                                              load_from_cache_file=False,
                                              keep_in_memory=True)
            preprocessed_train = sampled_train.map(self.ntp_model.tokenize,
                                                   remove_columns=sampled_train.column_names,
                                                   load_from_cache_file=False,
                                                   keep_in_memory=True)
            preprocessed_train.set_format("torch")

            pbar = tqdm(preprocessed_train.iter(batch_size=self.batch_size),
                        total=total_n_batch)

            train_loss = 0

            optimizer = self.ntp_model.get_suggested_optimizer()

            for i, batch in enumerate(pbar, start=1):

                optimizer.zero_grad()

                prepared_input = self.ntp_model.prepare_input(batch)
                output, loss = self.ntp_model.train_step(prepared_input)

                loss.backward()
                optimizer.step()

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
                    self.ntp_model.save(self.output_path)

                else:
                    no_change_counter += 1

    def validation(self, preprocessed_validation: datasets.Dataset):

        print("VALIDATION")
        self.ntp_model.eval()

        # ceil because we don't drop the last batch
        total_n_batch = ceil(preprocessed_validation.num_rows / self.eval_batch_size)

        pbar_val = tqdm(preprocessed_validation.iter(batch_size=self.eval_batch_size),
                        total=total_n_batch)

        val_loss = 0
        matches = 0

        for i, batch in enumerate(pbar_val, start=1):

            prepared_input = self.ntp_model.prepare_input(batch)
            acc, loss = self.ntp_model.valid_step(prepared_input)

            val_loss += loss.item()
            matches += acc

            # we update the loss every 1% progress considering the total n° of batches
            if (i % ceil(total_n_batch / 100)) == 0:
                pbar_val.set_description(f"Val Loss -> {(val_loss / i):.6f}")

        print(matches / preprocessed_validation.num_rows)

        pbar_val.close()

        return val_loss / len(pbar_val)

    def evaluate(self, test_dataset: datasets.Dataset):
        self.ntp_model.eval()
        preprocessed_test = test_dataset.map(self.ntp_model.tokenize,
                                             remove_columns=test_dataset.column_names)
        preprocessed_test.set_format("torch")

        total_n_batch = ceil(preprocessed_test.num_rows / self.eval_batch_size)

        pbar_test = tqdm(preprocessed_test.iter(batch_size=self.eval_batch_size),
                         total=total_n_batch)

        matches = 0

        for i, batch in enumerate(pbar_test, start=1):

            prepared_input = self.ntp_model.prepare_input(batch)
            acc, _ = self.ntp_model.valid_step(prepared_input)

            matches += acc

            # we update the loss every 1% progress considering the total n° of batches
            if (i % ceil(total_n_batch / 100)) == 0:
                pbar_test.set_description(f"Acc -> {(matches / (i * self.eval_batch_size)):.6f}")

        acc = matches / preprocessed_test.num_rows
        return acc