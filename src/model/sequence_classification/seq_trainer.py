import os
from math import ceil
import itertools

import datasets
import numpy as np

from tqdm import tqdm
from transformers import T5TokenizerFast

from src import RANDOM_STATE, MODELS_DIR
from src.data.legal_dataset import LegalDataset
from src.model.clustering import ClusterLabelMapper, KMeansAlg
from src.model.lm.t5.flan_t5 import FineTunedFlanT5
from src.model.lm.t5.templates import ClusteredNTPSideInfo, DirectNTP, ClusteredNTP, DirectNTPSideInfo
from src.model.sentence_encoders import SentenceTransformerEncoder
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
        preprocessed_val = validation_dataset.map(lambda x: self.model.tokenize(x,
                                                                                self.cluster_label_mapper),
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

            self.model.train()

            # if no significant change happens to the loss after 10 epochs then early stopping
            if no_change_counter == 10:
                print("Early stopping")
                self.model.save_pretrained(self.output_path)
                break

            # at the start of each iteration, we randomly sample the train sequence and tokenize it
            sampled_train = train_dataset.map(LegalDataset.perform_sampling,
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
        self.model.eval()
        preprocessed_test = test_dataset.map(lambda x: self.model.tokenize(x,
                                                                           self.cluster_label_mapper),
                                             remove_columns=test_dataset.column_names)
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

        acc = matches / preprocessed_test.num_rows
        return acc

def flan_t5_main(n_epochs, batch_size, eval_batch_size, dataset, all_unique_labels, device):

    clus_alg = KMeansAlg(
        n_clusters=200,
        random_state=42,
        init="k-means++",
        n_init="auto"
    )

    sent_encoder = SentenceTransformerEncoder(
        device=device,
    )

    tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-small")
    model = FineTunedFlanT5.from_pretrained(
        "google/flan-t5-small",
        sentence_encoder=sent_encoder,
        all_labels=all_unique_labels,
        tokenizer=tokenizer,
        device=device,
        test_task=ClusteredNTPSideInfo()
    )

    new_words = ['<', '>']

    tokenizer.add_tokens(new_words)
    model.resize_token_embeddings(len(tokenizer))

    cluster_label = ClusterLabelMapper(sent_encoder, clus_alg)

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

    print("clustered ntp side info")
    trainer.model = FineTunedFlanT5.from_pretrained(trainer.output_path,
                                                    sentence_encoder=sent_encoder,
                                                    all_labels=all_unique_labels,
                                                    tokenizer=tokenizer,
                                                    device=device,
                                                    test_task=ClusteredNTPSideInfo())


    trainer.evaluate(test)

    print("direct ntp")
    trainer.model.set_test_task(DirectNTP())
    trainer.evaluate(test)

    print("direct ntp with side info")
    trainer.model.set_test_task(DirectNTPSideInfo())
    trainer.evaluate(test)

    print("clustered ntp")
    trainer.model.set_test_task(ClusteredNTP())
    trainer.evaluate(test)


if __name__ == "__main__":
    seed_everything(RANDOM_STATE)

    # PARAMETERS
    n_epochs = 20
    batch_size = 4
    eval_batch_size = 2
    device = "cuda:0"

    ds = LegalDataset.load_dataset()
    dataset_dict = ds.get_hf_datasets()
    all_unique_labels = ds.all_unique_labels

    flan_t5_main(n_epochs, batch_size, eval_batch_size, dataset_dict, all_unique_labels, device)
