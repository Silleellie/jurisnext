import os
from math import ceil
import itertools

import gc
from typing import Optional

import torch
import datasets
import numpy as np
from sklearn.utils import compute_class_weight

from tqdm import tqdm

from src import RANDOM_STATE, MODELS_DIR
from src.data.legal_dataset import LegalDataset
from src.model.clustering import ClusterLabelMapper, KMeansAlg
from src.model.next_title_prediction.ntp_models import NTPFlanT5, BoolNTP
from src.model.sentence_encoders import SentenceTransformerEncoder
from src.model.next_title_prediction.ntp_models.bert import NTPBert
from src.model.next_title_prediction.ntp_models.multimodal import MultimodalFusionForSequenceClassification, \
    MultimodalFusionConfig
from src.model.next_title_prediction.ntp_models.multimodal.fusion import NTPMultimodalFusion
from src.model.next_title_prediction.ntp_models.nli_deberta import NTPNliDeberta
from src.model.next_title_prediction.ntp_models.lm import DirectNTP, DirectNTPSideInfo, \
    ClusteredNTP, ClusteredNTPSideInfo
from src.model.next_title_prediction.ntp_models_interface import NTPModel
from src.utils import seed_everything


class SeqTrainer:

    def __init__(self,
                 n_epochs: int,
                 batch_size: int,
                 ntp_model: NTPModel,
                 all_labels: list,
                 device: str = 'cuda:0',
                 eval_batch_size: Optional[int] = None,
                 num_workers: int = 4,
                 output_name: Optional[str] = None):

        self.ntp_model = ntp_model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.all_labels = all_labels
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers
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


def flan_t5_main(n_epochs, batch_size, eval_batch_size, device="cuda:0", use_cluster_alg=True):
    ds = LegalDataset.load_dataset()
    dataset = ds.get_hf_datasets()
    all_unique_labels = ds.all_unique_labels

    cluster_label = None

    train_tasks = [
        DirectNTP(),
        DirectNTPSideInfo(),
        BoolNTP(list(all_unique_labels))
    ]

    sent_encoder = SentenceTransformerEncoder(
        device=device,
    )

    test_task = DirectNTPSideInfo()

    if use_cluster_alg:

        clus_alg = KMeansAlg(
            n_clusters=200,
            random_state=42,
            init="k-means++",
            n_init="auto"
        )

        cluster_label = ClusterLabelMapper(sent_encoder, clus_alg)
        test_task = ClusteredNTPSideInfo()
        train_tasks.extend([ClusteredNTP(), ClusteredNTPSideInfo()])

    model_ntp = NTPFlanT5(
        "google/flan-t5-small",
        sentence_encoder=sent_encoder,
        cluster_label_mapper=cluster_label,

        training_tasks=train_tasks,
        test_task=test_task,
        all_unique_labels=all_unique_labels,
        device='cuda:0'
    )

    new_words = ['<', '>']

    model_ntp.tokenizer.add_tokens(new_words)
    model_ntp.model.resize_token_embeddings(len(model_ntp.tokenizer))

    trainer = SeqTrainer(
        ntp_model=model_ntp,
        n_epochs=n_epochs,
        batch_size=batch_size,
        all_labels=all_unique_labels,
        eval_batch_size=eval_batch_size,
    )

    train = dataset["train"]
    val = dataset["validation"]
    test_list = dataset["test"]

    trainer.train(train, val)

    gc.collect()
    torch.cuda.empty_cache()

    print("EVALUATION")
    trainer.model = NTPFlanT5.load(trainer.output_path)

    # check which task yield better results
    for task in train_tasks:

        print(f"Evaluating task {repr(task)}")
        trainer.model.set_test_task(task)
        all_acc = []
        for i, test in enumerate(test_list):
            print(f"Eval on {i}-th test set")
            acc = trainer.evaluate(test)
            all_acc.append(acc)

        print(np.mean(all_acc))


def multimodal_main(n_epochs, batch_size, eval_batch_size, device="cuda:0", use_cluster_alg=True):
    ds = LegalDataset.load_dataset()
    dataset = ds.get_hf_datasets()
    all_unique_labels = ds.all_unique_labels

    cluster_label = None

    if use_cluster_alg:

        clus_alg = KMeansAlg(
            n_clusters=50,
            random_state=42,
            init="k-means++",
            n_init="auto"
        )

        sent_encoder = SentenceTransformerEncoder(
            device=device,
        )

        cluster_label = ClusterLabelMapper(sent_encoder, clus_alg)

    train = dataset["train"]
    val = dataset["validation"]
    test_list = dataset["test"]

    all_train_labels_occurrences = [y for x in train for y in x['title_sequence']]
    # "smoothing" so that a weight can be calculated for labels which do not appear in the
    # train set
    all_train_labels_occurrences.extend(all_unique_labels)

    labels_weights = compute_class_weight(class_weight='balanced',
                                          classes=np.unique(all_unique_labels),
                                          y=all_train_labels_occurrences)

    model = MultimodalFusionForSequenceClassification(
        MultimodalFusionConfig(
            image_encoder_params={
                "input_dims": [1, 32, 64, 128, 64, 10],
                "output_dims": [32, 64, 128, 64, 10, 5],
                "kernel_sizes": [7, 5, 5, 5, 5, 1]
            },
            text_encoder_params={
                "model_name": "nlpaueb/legal-bert-base-uncased",
                "model_hidden_states_num": 4,
                "hidden_size": 256,
                "directions_fusion_strat": "mean"
            },
            max_seq_len=100,
            label2id={x: i for i, x in enumerate(all_unique_labels)},
            id2label={i: x for i, x in enumerate(all_unique_labels)},
            labels_weights=labels_weights,
            device='cuda:0'
        ),
    )

    model_ntp = NTPMultimodalFusion(
        model=model,
        cluster_label_mapper=cluster_label,
    )

    trainer = SeqTrainer(
        ntp_model=model_ntp,
        n_epochs=n_epochs,
        batch_size=batch_size,
        all_labels=all_unique_labels,
        eval_batch_size=eval_batch_size,
        output_name=f"MultimodalFusion_{n_epochs}"
    )

    trainer.train(train, val)

    gc.collect()
    torch.cuda.empty_cache()

    print("EVALUATION")
    trainer.model = NTPMultimodalFusion.load(trainer.output_path)

    acc = []
    for test in test_list:
        acc.append(trainer.evaluate(test))
    print(np.mean(acc))


def bert_main(n_epochs, batch_size, eval_batch_size, device="cuda:0", use_cluster_alg=False):
    ds = LegalDataset.load_dataset()
    dataset = ds.get_hf_datasets()
    all_unique_labels = ds.all_unique_labels

    cluster_label = None

    if use_cluster_alg:

        clus_alg = KMeansAlg(
            n_clusters=50,
            random_state=42,
            init="k-means++",
            n_init="auto"
        )

        sent_encoder = SentenceTransformerEncoder(
            device=device,
        )

        cluster_label = ClusterLabelMapper(sent_encoder, clus_alg)

    train = dataset["train"]
    val = dataset["validation"]
    test_list = dataset["test"]

    all_train_labels_occurrences = [y for x in train for y in x['title_sequence']]
    # "smoothing" so that a weight can be calculated for labels which do not appear in the
    # train set
    all_train_labels_occurrences.extend(all_unique_labels)

    labels_weights = compute_class_weight(class_weight='balanced',
                                          classes=np.unique(all_unique_labels),
                                          y=all_train_labels_occurrences)

    ntp_model = NTPBert(
        'bert-base-uncased',
        cluster_label_mapper=cluster_label,

        problem_type="single_label_classification",
        num_labels=len(all_unique_labels),
        label2id={x: i for i, x in enumerate(all_unique_labels)},
        id2label={i: x for i, x in enumerate(all_unique_labels)},

        labels_weights=list(labels_weights),
        device='cuda:0'
    )

    trainer = SeqTrainer(
        ntp_model=ntp_model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        all_labels=all_unique_labels,
        eval_batch_size=eval_batch_size
    )

    trainer.train(train, val)

    gc.collect()
    torch.cuda.empty_cache()

    print("EVALUATION")
    trainer.model = NTPBert.load(trainer.output_path)

    acc = []
    for test in test_list:
        acc.append(trainer.evaluate(test))
    print(np.mean(acc))


def deberta_main(n_epochs, batch_size, eval_batch_size, device="cuda:0", use_cluster_alg=False):
    ds = LegalDataset.load_dataset()
    dataset = ds.get_hf_datasets()
    all_unique_labels = ds.all_unique_labels

    cluster_label = None

    if use_cluster_alg:
        clus_alg = KMeansAlg(
            n_clusters=50,
            random_state=42,
            init="k-means++",
            n_init="auto"
        )

        sent_encoder = SentenceTransformerEncoder(
            device=device,
        )

        cluster_label = ClusterLabelMapper(sent_encoder, clus_alg)

    train = dataset["train"]
    val = dataset["validation"]
    test_list = dataset["test"]

    all_train_labels_occurrences = [y for x in train for y in x['title_sequence']]
    # "smoothing" so that a weight can be calculated for labels which do not appear in the
    # train set
    all_train_labels_occurrences.extend(all_unique_labels)

    ntp_model = NTPNliDeberta(
        namepretrained_model_or_pth='cross-encoder/nli-deberta-v3-xsmall',
        all_unique_labels=list(all_unique_labels),
        cluster_label_mapper=cluster_label,
        device='cuda:0'
    )

    trainer = SeqTrainer(
        ntp_model=ntp_model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        all_labels=all_unique_labels,
        eval_batch_size=eval_batch_size
    )

    trainer.train(train, val)

    gc.collect()
    torch.cuda.empty_cache()

    print("EVALUATION")
    trainer.model = NTPNliDeberta.load(trainer.output_path)

    acc = []
    for test in test_list:
        acc.append(trainer.evaluate(test))
    print(np.mean(acc))


if __name__ == "__main__":

    seed_everything(RANDOM_STATE)

    # PARAMETERS
    n_epochs = 1
    batch_size = 2
    eval_batch_size = 1
    device = "cuda:0"

    flan_t5_main(n_epochs, batch_size, eval_batch_size, device)
    # multimodal_main(n_epochs, batch_size, eval_batch_size, device)
    # bert_main(n_epochs, batch_size, eval_batch_size, device)
    # deberta_main(n_epochs, batch_size, eval_batch_size, device)
