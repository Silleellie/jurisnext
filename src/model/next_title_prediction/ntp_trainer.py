import os
from math import ceil
import itertools

from typing import Optional

import datasets
import numpy as np
import wandb

from tqdm import tqdm

from src import MODELS_DIR
from src.data.legal_dataset import LegalDataset
from src.evaluation.metrics import Hit, Accuracy
from src.model.next_title_prediction.ntp_models_abtract import NTPModel


class NTPTrainer:

    def __init__(self,
                 n_epochs: int,
                 batch_size: int,
                 ntp_model: NTPModel,
                 all_labels: np.ndarray,
                 device: str = 'cuda:0',
                 eval_batch_size: Optional[int] = None,
                 output_name: Optional[str] = None,
                 log_wandb: bool = False):

        self.ntp_model = ntp_model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.all_labels = all_labels
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.device = device
        self.log_wandb = log_wandb

        # output name
        if output_name is None:
            # replace '/' with '_' to avoid creation of subdir (google/flan-t5-small -> google_flan-t5-small)
            output_name = f"{ntp_model.config.name_or_path.replace('/', '_')}_{n_epochs}"

        self.output_name = output_name
        self.output_path = os.path.join(MODELS_DIR, output_name)

    def train(self, train_dataset: datasets.Dataset, validation_dataset: datasets.Dataset = None):

        if self.ntp_model.cluster_label_mapper is not None:
            # retrieve all unique labels which appear in train set
            flat_train_labels = itertools.chain.from_iterable(train_dataset["title_sequence"])
            unique_train_labels = np.unique(np.fromiter(flat_train_labels, dtype=object)).astype(str)
            self.ntp_model._train_clusters(unique_train_labels, self.all_labels)

            if self.log_wandb:
                wandb.config.update({
                    "clusters": self.ntp_model.cluster_label_mapper.get_parameters()
                })

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
        min_delta = 1e-4

        train_step = 0
        val_step = 0
        for epoch in range(0, self.n_epochs):

            self.ntp_model.train()

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
                loss = self.ntp_model.train_step(prepared_input)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # we update the loss every 1% progress considering the total n° of batches
                if (i % ceil(total_n_batch / 100)) == 0:
                    train_step += 1
                    pbar.set_description(f"Epoch {epoch + 1}, Loss -> {(train_loss / i):.6f}")

                    if self.log_wandb:
                        wandb.log(
                            {'train/loss': (train_loss / i)},
                            step=train_step
                        )

            if self.log_wandb:
                wandb.log({"train/epoch": epoch + 1})

            pbar.close()

            if validation_dataset is not None:
                val_step, val_loss = self.validation(preprocessed_validation=preprocessed_val, val_step=val_step)

                # if there is a significant difference between the last minimum loss and the current one
                # set it as the new min loss and save the model parameters
                if (min_val_loss - val_loss) > min_delta:
                    min_val_loss = val_loss
                    self.ntp_model.save(self.output_path)

                    print(f"Validation loss is improved, model saved into {self.output_path}!")

        print(" Train completed! Check models saved into 'models' dir ".center(100, "*"))

    def validation(self, preprocessed_validation: datasets.Dataset, val_step: int):

        print("VALIDATION")
        self.ntp_model.eval()

        # ceil because we don't drop the last batch
        total_n_batch = ceil(preprocessed_validation.num_rows / self.eval_batch_size)

        pbar_val = tqdm(preprocessed_validation.iter(batch_size=self.eval_batch_size),
                        total=total_n_batch)

        val_loss = 0
        total_preds = []
        total_truths = []

        for i, batch in enumerate(pbar_val, start=1):

            prepared_input = self.ntp_model.prepare_input(batch)
            predictions, truths, loss = self.ntp_model.valid_step(prepared_input)

            val_loss += loss.item()

            total_preds.extend(predictions)
            total_truths.extend(truths)

            # we update the loss every 1% progress considering the total n° of batches
            if (i % ceil(total_n_batch / 100)) == 0:
                val_step += 1
                preds_so_far = np.array(total_preds)
                truths_so_far = np.array(total_truths)

                metric = Accuracy()
                if len(preds_so_far.squeeze().shape) > 1:
                    metric = Hit()

                result = metric(preds_so_far.squeeze(), truths_so_far)
                pbar_val.set_description(f"Val Loss -> {(val_loss / i):.6f}, "
                                         f"{metric} -> {result:.3f}")

                if self.log_wandb:
                    wandb.log(
                        {'val/loss': (val_loss / i),
                         f"val/{str(metric)}": result},

                        step=val_step
                    )

        pbar_val.close()

        # val_loss is computed for the entire batch, not for each sample, that's why is safe
        # to use pbar_val
        return val_loss / len(pbar_val), val_step
