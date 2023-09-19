import os
import time
from math import ceil

from typing import Optional, Callable, Literal

import datasets
import numpy as np
import wandb

from tqdm import tqdm

from src import MODELS_DIR
from src.evaluation.metrics import Hit, Accuracy, Metric
from src.model.next_title_prediction.ntp_models_abtract import NTPModel


class NTPTrainer:

    def __init__(self,
                 n_epochs: int,
                 batch_size: int,
                 ntp_model: NTPModel,
                 all_labels: np.ndarray,
                 train_sampling_fn: Callable,
                 device: str = 'cuda:0',
                 monitor_strategy: Literal['loss', 'metric'] = 'metric',
                 eval_batch_size: Optional[int] = None,
                 output_name: Optional[str] = None,
                 log_wandb: bool = False,
                 random_seed: Optional[int] = None):

        self.ntp_model = ntp_model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.all_labels = all_labels
        self.train_sampling_fn = train_sampling_fn
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.device = device
        self.log_wandb = log_wandb
        self.random_seed = random_seed
        self.monitor_strategy = monitor_strategy

        # output name
        if output_name is None:
            # replace '/' with '_' to avoid creation of subdir (google/flan-t5-small -> google_flan-t5-small)
            output_name = f"{ntp_model.config.name_or_path.replace('/', '_')}_{n_epochs}"

        self.output_name = output_name
        self.output_path = os.path.join(MODELS_DIR, output_name)

    def train(self, train_dataset: datasets.Dataset, validation_dataset: datasets.Dataset = None):

        # validation set remains the same and should NOT be sampled at each epoch, otherwise
        # results are not comparable
        self.ntp_model.eval()  # eval because the "tokenize" function select different tasks depending on the mode
        preprocessed_val = validation_dataset.map(self.ntp_model.tokenize,
                                                  remove_columns=validation_dataset.column_names,
                                                  load_from_cache_file=False,
                                                  desc="Tokenizing val set"
                                                  )
        preprocessed_val.set_format("torch")

        # depending on the monitor strategy, we want either this to decrease or to increase,
        # so we have a different initialization
        best_val_monitor_result = np.inf if self.monitor_strategy == "loss" else 0
        train_step = 0
        val_step = 0
        best_epoch = -1

        optimizer = self.ntp_model.get_suggested_optimizer()

        start = time.time()
        for epoch in range(0, self.n_epochs):

            self.ntp_model.train()

            # at the start of each iteration, we randomly sample the train sequence and tokenize it
            # Process the data in batches of 1 row
            # (we didn't invest time in vectorizing the sampling, so passing a higher batch size
            # has no effect)
            # We still use batched=True so that even if we augment data (lists are returned)
            # each augmented data is considered as a row
            shuffled_train = train_dataset.shuffle(seed=self.random_seed)
            sampled_train = shuffled_train.map(self.train_sampling_fn,
                                               batched=True,
                                               batch_size=1,
                                               remove_columns=train_dataset.column_names,
                                               load_from_cache_file=False,
                                               keep_in_memory=True,
                                               desc="Sampling train set with chosen strategy")
            preprocessed_train = sampled_train.map(self.ntp_model.tokenize,
                                                   remove_columns=sampled_train.column_names,
                                                   load_from_cache_file=False,
                                                   keep_in_memory=True,
                                                   desc="Tokenizing train set")
            preprocessed_train.set_format("torch")

            # ceil because we don't drop the last batch. It's here since if we are in
            # augment strategy, row number increases after preprocessing
            total_n_batch = ceil(preprocessed_train.num_rows / self.batch_size)

            pbar = tqdm(preprocessed_train.iter(batch_size=self.batch_size),
                        total=total_n_batch)

            train_loss = 0

            for i, batch in enumerate(pbar, start=1):

                optimizer.zero_grad()

                prepared_input = self.ntp_model.prepare_input(batch)
                loss = self.ntp_model.train_step(prepared_input)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # we update the loss every 1% progress considering the total n° of batches
                if (i % ceil(total_n_batch / 100)) == 0 or i == total_n_batch:
                    pbar.set_description(f"Epoch {epoch + 1}, Loss -> {(train_loss / i):.6f}")

                    if self.log_wandb:

                        to_log = {
                            'train/loss': (train_loss / i),
                            'train/step': train_step
                        }

                        if i == total_n_batch:
                            to_log['train/epoch'] = epoch + 1

                        wandb.log(to_log)

                    train_step += 1

            pbar.close()

            if validation_dataset is not None:
                val_result, val_step = self.validation(preprocessed_validation=preprocessed_val,
                                                       val_step=val_step,
                                                       epoch=epoch)

                if self.monitor_strategy == "loss":
                    monitor_str = "Val loss"
                    monitor_val = val_result["loss"]
                    should_save = monitor_val < best_val_monitor_result  # we want loss to decrease
                else:
                    metric_obj, monitor_val = val_result["metric"]
                    monitor_str = str(metric_obj)
                    should_save = monitor_val > best_val_monitor_result  # we want metric (acc/hit) to increase

                # we save the best model based on the reference metric result
                if should_save:
                    best_epoch = epoch + 1
                    best_val_monitor_result = monitor_val
                    self.ntp_model.save(self.output_path)

                    print(f"{monitor_str} improved, model saved into {self.output_path}!")

        if self.log_wandb:
            wandb.log({
                'train/best_model_epoch': best_epoch,
                'train/train time (sec)': int(time.time() - start) / 60  # we log minutes instead of secs
            })

        print(" Train completed! Check models saved into 'models' dir ".center(100, "*"))

    def validation(self, preprocessed_validation: datasets.Dataset, val_step: int, epoch: int):

        print("VALIDATION")
        self.ntp_model.eval()

        # ceil because we don't drop the last batch
        total_n_batch = ceil(preprocessed_validation.num_rows / self.eval_batch_size)

        pbar_val = tqdm(preprocessed_validation.iter(batch_size=self.eval_batch_size),
                        total=total_n_batch)

        metric: Metric = Accuracy()
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
            if (i % ceil(total_n_batch / 100)) == 0 or i == total_n_batch:
                preds_so_far = np.array(total_preds)
                truths_so_far = np.array(total_truths)

                if len(preds_so_far.squeeze().shape) > 1:
                    metric = Hit()

                result = metric(preds_so_far.squeeze(), truths_so_far)
                pbar_val.set_description(f"Val Loss -> {(val_loss / i):.6f}, "
                                         f"{metric} -> {result:.3f}")

                if self.log_wandb:

                    to_log = {'val/loss': (val_loss / i), f'val/{str(metric)}': result, 'val/step': val_step}

                    if i == total_n_batch:
                        to_log['val/epoch'] = epoch + 1

                    wandb.log(to_log)

                val_step += 1

        pbar_val.close()

        val_loss /= len(pbar_val)
        val_metric = metric(np.array(total_preds).squeeze(), np.array(total_truths))

        # val_loss is computed for the entire batch, not for each sample, that's why is safe
        # to use pbar_val
        return {"loss": val_loss, "metric": (metric, val_metric)}, val_step
