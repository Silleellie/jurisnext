import os
from collections import defaultdict
from math import ceil
import itertools

import datasets
import numpy
import numpy as np
from datasets import load_dataset
from sklearn.utils import compute_class_weight

from tqdm import tqdm
from transformers import AutoTokenizer

from src import RANDOM_STATE, ROOT_PATH, MODELS_DIR
from src.data.clustering import ClusterLabelMapper, KMeansAlg
from src.data.dataset_map_fn import sample_sequence
from src.model.sequence_classification.seq_models.bert import FineTunedBert
from src.model.sequence_classification.seq_models.nli_deberta import FineTunedNliDeberta
from src.sentence_encoders import SentenceEncoder, BertSentenceEncoder
from src.utils import seed_everything


class SeqTrainer:

    def __init__(self,
                 n_epochs,
                 batch_size,
                 model,
                 all_labels: np.ndarray,
                 cluster_label_mapper: ClusterLabelMapper = None,
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
        self.cluster_label_mapper = cluster_label_mapper

        # output name
        if output_name is None:
            output_name = f"{model.config.name_or_path}_{n_epochs}" if output_name is None else output_name

        self.output_path = os.path.join(MODELS_DIR, output_name)

        self.labels_map = None
        self.possible_labels_from_clusters = None

    def _train_clusters(self, train_dataset: datasets.Dataset):

        # retrieve all unique labels which appear in train set
        flat_train_labels = itertools.chain.from_iterable(train_dataset["title_sequence"])
        unique_train_labels = np.unique(np.fromiter(flat_train_labels, dtype=object)).astype(str)

        # fit the cluster label mapper with train labels and all labels which should be clustered (both are unique)
        self.cluster_label_mapper.fit(unique_train_labels, self.all_labels)

    def train(self, train_dataset: datasets.Dataset, validation_dataset: datasets.Dataset = None):

        if self.cluster_label_mapper is not None:
            self._train_clusters(train_dataset)

        # validation set remains the same and should NOT be sampled at each epoch, otherwise
        # results are not comparable
        self.model.eval()  # eval because the "tokenize" function select different tasks depending on the mode
        sampled_val = validation_dataset.map(sample_sequence,
                                             remove_columns=validation_dataset.column_names,
                                             load_from_cache_file=False)
        preprocessed_val = sampled_val.map(lambda x: self.model.tokenize(x,
                                                                         self.cluster_label_mapper),
                                           remove_columns=sampled_val.column_names,
                                           load_from_cache_file=False)
        preprocessed_val.set_format("torch")

        # ceil because we don't drop the last batch
        total_n_batch = ceil(train_dataset.num_rows / self.batch_size)

        # early stopping parameters
        min_val_loss = np.inf
        no_change_counter = 0
        min_delta = 1e-4

        for epoch in range(0, self.n_epochs):

            self.model.train()

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
            preprocessed_train = sampled_train.map(lambda x: self.model.tokenize(x,
                                                                                 self.cluster_label_mapper),
                                                   remove_columns=sampled_train.column_names,
                                                   load_from_cache_file=False,
                                                   keep_in_memory=True)
            preprocessed_train.set_format("torch")

            pbar = tqdm(preprocessed_train.iter(batch_size=self.batch_size),
                        total=total_n_batch)

            train_loss = 0

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
        preprocessed_test = sampled_test.map(lambda x: self.model.tokenize(x,
                                                                           self.cluster_label_mapper),
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
    n_epochs = 10
    batch_size = 2
    eval_batch_size = 1
    tokenizer_name = "cross-encoder/nli-deberta-v3-xsmall"

    dataset = load_dataset(os.path.join(ROOT_PATH, "src", "data", "hf_dataset_script"))

    all_labels_occurrences = np.array([el
                                       for split in dataset
                                       for element in dataset[split]
                                       for el in element["title_sequence"]])

    all_unique_labels = np.unique(all_labels_occurrences)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # labels_weights = compute_class_weight(class_weight='balanced', classes=all_unique_labels, y=all_labels_occurrences)
    #
    # model = FineTunedBert.from_pretrained(
    #     'bert-base-uncased',
    #     problem_type="single_label_classification",
    #     num_labels=len(all_unique_labels),
    #     id2label={idx: label for idx, label in
    #               enumerate(all_unique_labels)},
    #     label2id={label: idx for idx, label in
    #               enumerate(all_unique_labels)},
    #     labels_weights=labels_weights,
    #     tokenizer=tokenizer
    # ).to('cuda:0')

    clus_alg = KMeansAlg(
        n_clusters=50,
        random_state=42,
        init="k-means++",
        n_init="auto"
    )

    sent_encoder = BertSentenceEncoder(
        model_name="nlpaueb/legal-bert-base-uncased",
        token_fusion_strat="mean",
        hidden_states_fusion_strat="concat"
    )

    cluster_label = ClusterLabelMapper(sent_encoder, clus_alg)

    model = FineTunedNliDeberta.from_pretrained(
        "cross-encoder/nli-deberta-v3-xsmall",
        all_unique_labels=all_unique_labels,
        tokenizer=tokenizer,
    ).to('cuda:0')

    trainer = SeqTrainer(
        model=model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        all_labels=all_unique_labels,
        cluster_label_mapper=cluster_label,
        eval_batch_size=eval_batch_size,
    )

    train = dataset["train"]
    val = dataset["validation"]
    test = dataset["test"]

    trainer.train(train, val)

    trainer.model = FineTunedNliDeberta.from_pretrained(trainer.output_path,
                                                        all_unique_labels=all_unique_labels,
                                                        tokenizer=tokenizer)

    trainer.evaluate(test)
