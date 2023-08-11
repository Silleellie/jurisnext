import os
from math import ceil

import datasets
import numpy as np
import torch
from datasets import load_dataset
from sklearn.utils import compute_class_weight
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

from src import RANDOM_STATE, ROOT_PATH
from src.data.dataset_map_fn import sample_sequence
from src.utils import seed_everything


class BertTrainer:

    def __init__(self, n_epochs, batch_size,
                 model: str,
                 all_labels: np.ndarray,
                 labels_weights: np.ndarray,
                 device='cuda:0',
                 eval_batch_size=None, num_workers=4):

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = BertForSequenceClassification.from_pretrained(model,
                                                                   problem_type="single_label_classification",
                                                                   num_labels=len(all_labels),
                                                                   id2label={idx: label for idx, label in
                                                                             enumerate(all_labels)},
                                                                   label2id={label: idx for idx, label in
                                                                             enumerate(all_labels)}
                                                                   ).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.all_labels = all_labels
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.device = device

        # labels weights are set since dataset is unbalanced
        self.labels_weights = torch.from_numpy(labels_weights).to(device).to(torch.float32)

        self.optimizer = torch.optim.AdamW(list(self.model.parameters()), lr=2e-5)

    def tokenize(self, x):

        output = self.tokenizer(', '.join(x["input_title_sequence"]),
                                return_tensors='pt',
                                max_length=512,
                                truncation=True,
                                padding='max_length')

        labels = torch.Tensor([self.model.config.label2id[x["immediate_next_title"]]])

        return {'input_ids': output['input_ids'].squeeze().to(self.device),
                'token_type_ids': output['token_type_ids'].squeeze().to(self.device),
                'attention_mask': output['attention_mask'].squeeze().to(self.device),
                'labels': labels.to(self.device).long()}

    def train(self, train_dataset: datasets.Dataset, validation_dataset: datasets.Dataset = None):

        # validation set remains the same and should NOT be sampled at each epoch, otherwise
        # results are not comparable
        self.model.eval()  # eval because the "tokenize" function select different tasks depending on the mode
        sampled_val = validation_dataset.map(sample_sequence,
                                             remove_columns=validation_dataset.column_names,
                                             load_from_cache_file=False)
        preprocessed_val = sampled_val.map(self.tokenize, remove_columns=sampled_val.column_names)
        preprocessed_val.set_format("torch")
        self.model.train()

        # ceil because we don't drop the last batch
        total_n_batch = ceil(train_dataset.num_rows / self.batch_size)

        loss_fct = CrossEntropyLoss(weight=self.labels_weights)

        # early stopping parameters
        min_val_loss = np.inf
        no_change_counter = 0
        min_delta = 1e-4

        for epoch in range(0, self.n_epochs):

            # if no significant change happens to the loss after 10 epochs then early stopping
            if no_change_counter == 10:
                print("Early stopping")
                self.model.load_state_dict(torch.load('bert.pt'))
                break

            # at the start of each iteration, we randomly sample the train sequence and tokenize it
            sampled_train = train_dataset.map(sample_sequence,
                                              remove_columns=train_dataset.column_names,
                                              load_from_cache_file=False,
                                              keep_in_memory=True)
            preprocessed_train = sampled_train.map(self.tokenize,
                                                   remove_columns=sampled_train.column_names,
                                                   load_from_cache_file=False,
                                                   keep_in_memory=True)
            preprocessed_train.set_format("torch")

            pbar = tqdm(preprocessed_train.iter(batch_size=self.batch_size),
                        total=total_n_batch)

            train_loss = 0
            self.model.train()
            for i, batch in enumerate(pbar, start=1):

                self.optimizer.zero_grad()

                labels = batch['labels'].to(self.device)
                batch = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}

                output = self.model(**batch)
                loss = loss_fct(output.logits.view(-1, self.model.config.num_labels), labels.view(-1))
                train_loss += loss.item()

                loss.backward()
                self.optimizer.step()

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
                    torch.save(self.model.state_dict(), 'bert.pt')

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

        loss_fct = CrossEntropyLoss(weight=self.labels_weights)

        for i, batch in enumerate(pbar_val, start=1):

            with torch.no_grad():

                labels = batch['labels'].to(self.device)
                batch = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}

                output = self.model(**batch)

                loss = loss_fct(output.logits.view(-1, self.model.config.num_labels), labels.view(-1))
                val_loss += loss.item()

                predictions = output.logits.argmax(1)
                truth = labels.view(-1)

                matches += sum(
                    x == y for x, y in zip(predictions, truth)
                )

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
        preprocessed_test = sampled_test.map(self.tokenize,
                                             remove_columns=sampled_test.column_names)
        preprocessed_test.set_format("torch")

        total_n_batch = ceil(preprocessed_test.num_rows / self.eval_batch_size)

        pbar_test = tqdm(preprocessed_test.iter(batch_size=self.eval_batch_size),
                         total=total_n_batch)

        matches = 0

        for i, batch in enumerate(pbar_test, start=1):

            with torch.no_grad():

                labels = batch['labels'].to(self.device)
                batch = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}

                output = self.model(**batch)

                predictions = output.logits.argmax(1)
                truth = labels.view(-1)

                matches += sum(
                    x == y for x, y in zip(predictions, truth)
                )

            # we update the loss every 1% progress considering the total n° of batches
            if (i % ceil(total_n_batch / 100)) == 0:
                pbar_test.set_description(f"Acc -> {(matches / (i * self.eval_batch_size)):.6f}")

        print(matches / preprocessed_test.num_rows)


if __name__ == "__main__":
    seed_everything(RANDOM_STATE)

    # PARAMETERS
    dataset = load_dataset(os.path.join(ROOT_PATH, "src", "data", "hf_dataset_script"))

    all_labels_occurrences = np.array([el
                                       for split in dataset
                                       for element in dataset[split]
                                       for el in element["title_sequence"]])

    all_unique_labels = np.unique(all_labels_occurrences)

    labels_weights = compute_class_weight(class_weight='balanced', classes=all_unique_labels, y=all_labels_occurrences)

    train = dataset["train"]
    val = dataset["validation"]
    test = dataset["test"]

    trainer = BertTrainer(
        model='bert-base-uncased',
        n_epochs=100,
        batch_size=2,
        all_labels=all_unique_labels,
        labels_weights=labels_weights,
        eval_batch_size=2
    )

    trainer.train(train, val)

    trainer.model.load_state_dict(torch.load('bert.pt'))

    trainer.evaluate(test)
