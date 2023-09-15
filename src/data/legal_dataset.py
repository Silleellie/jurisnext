import re
import os
import pickle
import random
from collections import namedtuple
from functools import cached_property
from pathlib import Path
from typing import Tuple, Dict, Literal

import datasets
import numpy as np
import pandas as pd
import wandb
from datasets import Dataset, NamedSplit
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

from src import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, ExperimentConfig, REPORTS_DIR


def clean_original_dataset(original_dataset: pd.DataFrame):
    # we pick only useful columns from model training
    cleaned_dataset = original_dataset[["case:concept:name",
                                        "concept:name",
                                        "concept:name:paragraphs",
                                        "concept:name:section"]]

    # we rename columns to make them more readable
    cleaned_dataset = cleaned_dataset.rename(columns={"case:concept:name": "case_id",
                                                      "concept:name": "title",
                                                      "concept:name:paragraphs": "text",
                                                      "concept:name:section": "rel_keywords"})

    # from list of sentences to single paragraph with \n separator
    cleaned_dataset["text"] = cleaned_dataset["text"].apply("\n".join)
    # some documents have no paragraph information, we explicitate this
    cleaned_dataset["text"] = cleaned_dataset["text"].replace("", "!!No paragraph content!!")

    # some elements have [None] as concept:name:section, we substitute them with []
    cleaned_dataset["rel_keywords"] = cleaned_dataset["rel_keywords"].apply(
        lambda x: [] if all(el is None for el in x) else x
    )
    # rel keywords of the paragraph are grouped together
    cleaned_dataset["rel_keywords"] = cleaned_dataset["rel_keywords"].apply(", ".join)
    # some documents have no paragraph information (thus no relevant keywords), we explicitate this
    cleaned_dataset["rel_keywords"] = cleaned_dataset["rel_keywords"].replace("", "!!No paragraph content!!")

    cleaned_dataset["title"] = cleaned_dataset["title"].apply(lambda x: re.sub(r"<\s*org[^>]", r"<org>", x))
    cleaned_dataset["title"] = cleaned_dataset["title"].apply(lambda x: re.sub(r"<\s*person[^>]", r"<person>", x))
    cleaned_dataset["title"] = cleaned_dataset["title"].apply(lambda x: re.sub(r"<\s*law[^>]", r"<law>", x))

    cleaned_dataset["title"] = cleaned_dataset["title"].apply(lambda x: re.sub(r"_person >", r"<person>", x))

    # appellatives start of sentence
    cleaned_dataset["title"] = cleaned_dataset["title"].apply(
        lambda x: re.sub(r'^(mr|ms)(?=\s+|$)', '<appellative>', x))

    # appellatives middle of sentence
    cleaned_dataset["title"] = cleaned_dataset["title"].apply(
        lambda x: re.sub(r'\s+(mr|ms)(?=\s+|$)', '<appellative>', x))

    # add space when missing around <>
    cleaned_dataset["title"] = cleaned_dataset["title"].apply(lambda x: re.sub(r'(?<=\S)<([^>]+)>', r' <\g<1>>', x))

    # add space when missing around <>
    cleaned_dataset["title"] = cleaned_dataset["title"].apply(lambda x: re.sub(r'<([^>]+)>(?=\S)', r'<\g<1>> ', x))

    # dh 1 -> dh1 (they are both in the dataset)
    cleaned_dataset["title"] = cleaned_dataset["title"].apply(lambda x: re.sub(r'dh(\d+)', r'dh \g<1>', x))

    # caseNUMBER -> case NUMBER
    cleaned_dataset["title"] = cleaned_dataset["title"].apply(lambda x: re.sub(r'case(\d+)', r'case \g<1>', x))

    # numbers
    cleaned_dataset["title"] = cleaned_dataset["title"].apply(lambda x: re.sub(r'(?<=\s)[ct]?\d+', '<number>', x))

    return cleaned_dataset


def clean_keywords(original_dataset: pd.DataFrame):
    nltk.download('stopwords')
    stops = pd.Series(stopwords.words('english'))

    cleaned_keywords_col = original_dataset["rel_keywords"].str.split(", ")

    # remove stopwords
    cleaned_keywords_col = cleaned_keywords_col.explode()
    only_stopwords_index = cleaned_keywords_col.isin(stops)
    cleaned_keywords_col = cleaned_keywords_col[~only_stopwords_index]
    cleaned_keywords_col = cleaned_keywords_col.groupby(level=0).agg(list)

    # in case all keywords had stopwords, we have removed them all, so we need to add a placeholder text
    cleaned_keywords_col[cleaned_keywords_col.str.len() == 0] = ["!!No paragraph content!!"]
    original_dataset["rel_keywords"] = cleaned_keywords_col.str.join(", ")

    return original_dataset


def max_ngram_cut(cleaned_dataset: pd.DataFrame, cutoff_ngram: int = None):
    tokenizer_pattern = r"<[^>]+>|\S+"

    ngram_cut_df = cleaned_dataset.copy()
    ngram_cut_df["title"] = ngram_cut_df["title"].apply(lambda x:
                                                        ' '.join(re.findall(tokenizer_pattern, x)[:cutoff_ngram]))

    return ngram_cut_df


SeqTargetTuple = namedtuple("SeqTargetTuple", ["seq_title", "target_title",
                                               "seq_text", "target_text",
                                               "seq_keywords", "target_keywords"])


class LegalDataset:
    cleaned_dataset_path: str = os.path.join(INTERIM_DATA_DIR, "cleaned_dataframe.pkl")
    train_path: str = os.path.join(PROCESSED_DATA_DIR, "train.pkl")
    val_path: str = os.path.join(PROCESSED_DATA_DIR, "validation.pkl")
    test_list_path: str = os.path.join(PROCESSED_DATA_DIR, "test_list.pkl")

    def __init__(self,
                 n_test_set: int,
                 random_seed: int,
                 sampling_strategy: Literal['random', 'augment'] = "random"):

        self.random_seed = random_seed
        self.sampling_strategy = sampling_strategy
        self.train_df, self.val_df, self.test_df_list = self._generate_splits_and_sample(n_test_set)

    @cached_property
    def all_unique_labels(self) -> np.ndarray[str]:
        all_labels = self.train_df["title_sequence"].explode().tolist()
        all_labels.extend(
            self.val_df["input_title_sequence"].explode().tolist() +
            self.val_df["immediate_next_title"].tolist()
        )
        for test_df in self.test_df_list:
            all_labels.extend(
                test_df["input_title_sequence"].explode().tolist() +
                test_df["immediate_next_title"].tolist()
            )

        return pd.unique(np.array(all_labels))

    @cached_property
    def all_ner_tokens(self) -> np.ndarray[str]:

        all_labels = pd.Series(self.all_unique_labels)
        all_ner_tokens = np.unique(all_labels.str.extractall(r"(?P<ner_token>(<[^>]+>)"))

        return np.unique(all_ner_tokens)

    def _generate_splits_and_sample(self, n_test_set: int):

        print("Creating dataset splits...")

        # remove the splits to have a fresh start
        Path(self.train_path).unlink(missing_ok=True)
        Path(self.val_path).unlink(missing_ok=True)
        Path(self.test_list_path).unlink(missing_ok=True)

        cleaned_dataset: pd.DataFrame = pd.read_pickle(self.cleaned_dataset_path)
        train_dataset, val_dataset, test_dataset = self._split_dataset(cleaned_dataset)

        train_dataset = self._group_dataset(train_dataset, to_sample=False)
        val_dataset = self._group_dataset(val_dataset, to_sample=True)
        test_dataset = [self._group_dataset(test_dataset, to_sample=True) for _ in range(n_test_set)]

        train_dataset.to_pickle(self.train_path)
        val_dataset.to_pickle(self.val_path)

        with open(self.test_list_path, "wb") as f:
            pickle.dump(test_dataset, f)

        return train_dataset, val_dataset, test_dataset

    def _split_dataset(self, cleaned_dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        unique_case_ids = list(pd.unique(cleaned_dataset["case_id"]))

        train_ids, test_ids = train_test_split(
            unique_case_ids,
            test_size=0.2,
            random_state=self.random_seed,
            shuffle=True
        )

        train_ids, val_ids = train_test_split(
            train_ids,
            test_size=0.1,
            random_state=self.random_seed,
            shuffle=True
        )

        train_set = cleaned_dataset[cleaned_dataset['case_id'].isin(train_ids)]
        val_set = cleaned_dataset[cleaned_dataset['case_id'].isin(val_ids)]
        test_set = cleaned_dataset[cleaned_dataset['case_id'].isin(test_ids)]

        return train_set, val_set, test_set

    def _group_dataset(self, dataset_split, to_sample: bool = False):

        grouped_split = dataset_split.groupby("case_id")[['title', 'text', 'rel_keywords']].agg(list)

        # dataset has been grouped by case id, so now we have sequences of titles, texts, rel keywords
        grouped_split = grouped_split.rename(columns={
            'title': 'title_sequence',
            'text': 'text_sequence',
            'rel_keywords': 'rel_keywords_sequence'
        })

        grouped_split = grouped_split.reset_index()

        if to_sample:
            sampled_split = grouped_split.apply(self.perform_sampling, axis=1)
            grouped_split = pd.DataFrame.from_records(sampled_split)
            if self.sampling_strategy == "augment":
                grouped_split = grouped_split.explode(column=grouped_split.columns.tolist())

        return grouped_split

    def perform_sampling(self, batch):
        if self.sampling_strategy == "random":
            return self._sample_sequences(batch)
        else:
            return self._augment_sequences(batch)

    @staticmethod
    def _sample_sequences(batch):

        # a sequence has at least 1 data point, but it can have more depending on the length of the sequence
        # We must ensure that at least an element can be used as test set
        # in the "sliding_training_size" is included the target item
        sliding_size = random.randint(1, len(batch["text_sequence"]) - 1)

        # TO DO: consider starting always from the initial paragraph,
        # rather than varying the starting point of the seq
        # start_index = random.randint(0, len(batch["text_sequence"]) - sliding_size - 1)
        start_index = 0
        end_index = start_index + sliding_size

        return {
            "case_id": batch["case_id"],
            "input_title_sequence": batch["title_sequence"][start_index:end_index],
            "input_text_sequence": batch["text_sequence"][start_index:end_index],
            "input_keywords_sequence": batch["rel_keywords_sequence"][start_index:end_index],
            "immediate_next_title": batch["title_sequence"][end_index],
            "immediate_next_text": batch["text_sequence"][end_index],
            "immediate_next_rel_keywords": batch["rel_keywords_sequence"][end_index]
        }

    @staticmethod
    def _augment_sequences(sample):

        all_title_sequence = sample["title_sequence"]
        all_text_sequence = sample["text_sequence"]
        all_keywords_sequence = sample["rel_keywords_sequence"]

        assert len(all_title_sequence) >= 2, "All sequences must have at least 2 data points"

        n_sequences = len(all_title_sequence)

        all_seq = []
        for i in range(1, n_sequences):
            seq_title = all_title_sequence[0:i]
            seq_text = all_text_sequence[0:i]
            seq_keywords = all_keywords_sequence[0:i]

            target_title = all_title_sequence[i]
            target_text = all_text_sequence[i]
            target_keywords = all_keywords_sequence[i]

            all_seq.append(SeqTargetTuple(seq_title, target_title,
                                          seq_text, target_text,
                                          seq_keywords, target_keywords))

        return {
            "case_id": [sample["case_id"] for _ in range(1, n_sequences)],
            "input_title_sequence": [el.seq_title for el in all_seq],
            "input_text_sequence": [el.seq_text for el in all_seq],
            "input_keywords_sequence": [el.seq_keywords for el in all_seq],
            "immediate_next_title": [el.target_title for el in all_seq],
            "immediate_next_text": [el.target_text for el in all_seq],
            "immediate_next_rel_keywords": [el.target_keywords for el in all_seq]
        }

    def get_hf_datasets(self, merge_train_val: bool = False) -> Dict[str, datasets.Dataset]:

        val_hf_dataset = None
        if merge_train_val is True:
            cleaned_dataset: pd.DataFrame = pd.read_pickle(self.cleaned_dataset_path)
            cleaned_grouped = self._group_dataset(cleaned_dataset)

            rows_to_add_train = cleaned_grouped[cleaned_grouped["case_id"].isin(self.val_df["case_id"])]
            self.train_df = pd.concat([self.train_df, rows_to_add_train])
        else:
            val_hf_dataset = Dataset.from_pandas(self.val_df, split=datasets.Split.VALIDATION, preserve_index=False)

        train_hf_ds = Dataset.from_pandas(self.train_df, split=datasets.Split.TRAIN, preserve_index=False)
        test_hf_ds_list = [
            Dataset.from_pandas(test_df, split=NamedSplit(f"test_{i}"), preserve_index=False)
            for i, test_df in enumerate(self.test_df_list)
        ]

        # we create a dataset dict containing each split
        dataset_dict = {
            "train": train_hf_ds
        }
        if val_hf_dataset is not None:
            dataset_dict["validation"] = val_hf_dataset

        dataset_dict["test"] = test_hf_ds_list

        return dataset_dict

    @classmethod
    def load_dataset(cls, exp_config: ExperimentConfig):
        obj = cls.__new__(cls)  # Does not call __init__
        super(LegalDataset, obj).__init__()  # Don't forget to call any polymorphic base class initializers

        print("Loading dataset splits...")

        if any([
            not os.path.isfile(cls.train_path),
            not os.path.isfile(cls.val_path),
            not os.path.isfile(cls.test_list_path)
        ]):
            raise FileNotFoundError("Some splits are missing, the dataset can't be loaded! "
                                    "Instantiate this class with the proper constructor the first time you use it!")

        obj.train_df = pd.read_pickle(cls.train_path)
        obj.val_df = pd.read_pickle(cls.val_path)

        with open(cls.test_list_path, "rb") as f:
            obj.test_df_list = pickle.load(f)

        obj.random_seed = exp_config.random_seed
        obj.sampling_strategy = exp_config.seq_sampling_strategy

        return obj


def plot_save_label_counts(ds_frame: pd.DataFrame, fig_path: str):
    ax = ds_frame.head(30).plot.barh(x='titles', y='count', rot=0, figsize=(10.4, 8))

    # so that top elements are in the upper part of the diagram
    ax.invert_yaxis()

    plot_x_index_ticks = plt.xticks()[0][1:-1]

    for tick in plot_x_index_ticks:
        ax.axvline(x=tick, ls=':', color='tab:orange', zorder=0)

    plt.tight_layout()

    plt.savefig(fig_path)

    plt.clf()
    plt.cla()
    plt.close()


def log_parameters(ds: LegalDataset, exp_name: str):
    parameters_to_log = {
        "data/train_set_n_cases": ds.train_df.shape[0],
        "data/val_set_n_cases": ds.val_df.shape[0],
        "data/test_set_n_cases": ds.test_df_list[0].shape[0],
        "data/original_title_distribution_plot": wandb.Image(os.path.join(REPORTS_DIR,
                                                                          "data_plots",
                                                                          exp_name,
                                                                          "original_titles_counts.png"),
                                                             mode="RGB"),
        "data/train_distribution_plot": wandb.Image(os.path.join(REPORTS_DIR,
                                                                 "data_plots",
                                                                 exp_name,
                                                                 "train_titles_counts.png"),
                                                    mode="RGB"),
        "data/val_distribution_plot": wandb.Image(os.path.join(REPORTS_DIR,
                                                               "data_plots",
                                                               exp_name,
                                                               "val_titles_counts.png"),
                                                  mode="RGB"),
        "data/test_distribution_plot": [wandb.Image(os.path.join(REPORTS_DIR,
                                                                 "data_plots",
                                                                 exp_name,
                                                                 "test_sets",
                                                                 f"test_{test_split_idx}_titles_counts.png"),
                                                    mode="RGB",
                                                    caption=f"Test idx: {test_split_idx}")
                                        for test_split_idx in range(len(ds.test_df_list))]
    }

    wandb.log(parameters_to_log)


def data_main(exp_config: ExperimentConfig):
    if not os.path.isfile(os.path.join(RAW_DATA_DIR, "pre-processed_representations.pkl")):
        raise FileNotFoundError("Please add 'pre-processed_representations.pkl' into 'data/raw' folder!")

    original_df_path = os.path.join(RAW_DATA_DIR, "pre-processed_representations.pkl")
    cleaned_df_output_path = os.path.join(INTERIM_DATA_DIR, "cleaned_dataframe.pkl")

    original_df: pd.DataFrame = pd.read_pickle(original_df_path)
    cleaned_df = clean_original_dataset(original_df)

    if exp_config.remove_stopwords_kwds is True:
        cleaned_df = clean_keywords(cleaned_df)

    ngram_cut_df = max_ngram_cut(cleaned_df, cutoff_ngram=exp_config.ngram_label)
    ngram_cut_df.to_pickle(cleaned_df_output_path)

    # the constructor will create and dump the splits
    ds = LegalDataset(n_test_set=exp_config.n_test_set,
                      random_seed=exp_config.random_seed,
                      sampling_strategy=exp_config.seq_sampling_strategy)

    # create directory where all the distributions will be saved
    os.makedirs(os.path.join(REPORTS_DIR, "data_plots", exp_config.exp_name), exist_ok=True)

    # PLOT ORIGINAL DATASET TITLES DISTRIBUTION
    cleaned_df_to_plot = ngram_cut_df.rename(columns={"title": "titles"})
    labels_count = cleaned_df_to_plot["titles"].value_counts().reset_index()
    labels_count["titles"] = labels_count["titles"].apply(lambda x: x[:20] + "..." if len(x) > 20 else x)

    plot_save_label_counts(labels_count, os.path.join(REPORTS_DIR, "data_plots", exp_config.exp_name,
                                                      "original_titles_counts.png"))

    # check that all test sets have the same number of cases
    assert all(ds.test_df_list[0].shape[0] == test_set.shape[0] for test_set in ds.test_df_list)

    # PLOT TRAIN TITLES DISTRIBUTION
    train_df_to_plot = ds.train_df.explode("title_sequence").rename(columns={"title_sequence": "titles"})
    labels_count = train_df_to_plot["titles"].value_counts().reset_index()
    labels_count["titles"] = labels_count["titles"].apply(lambda x: x[:20] + "..." if len(x) > 20 else x)

    plot_save_label_counts(labels_count, os.path.join(REPORTS_DIR, "data_plots", exp_config.exp_name,
                                                      "train_titles_counts.png"))

    # PLOT VAL TITLES DISTRIBUTION
    val_df_to_plot = ds.val_df.explode("input_title_sequence")
    labels_count = pd.concat((val_df_to_plot["input_title_sequence"],
                              val_df_to_plot["immediate_next_title"])).value_counts().reset_index()
    labels_count = labels_count.rename(columns={"index": "titles"})
    labels_count["titles"] = labels_count["titles"].apply(lambda x: x[:20] + "..." if len(x) > 20 else x)

    plot_save_label_counts(labels_count, os.path.join(REPORTS_DIR, "data_plots", exp_config.exp_name,
                                                      "val_titles_counts.png"))

    # PLOT TEST TITLES DISTRIBUTION

    # create directory where all the test sets distributions will be saved
    os.makedirs(os.path.join(REPORTS_DIR, "data_plots", exp_config.exp_name, "test_sets"), exist_ok=True)

    for i, test_df in enumerate(ds.test_df_list):
        test_df_to_plot = test_df.explode("input_title_sequence")
        labels_count = pd.concat((test_df_to_plot["input_title_sequence"],
                                  test_df_to_plot["immediate_next_title"])).value_counts().reset_index()
        labels_count = labels_count.rename(columns={"index": "titles"})
        labels_count["titles"] = labels_count["titles"].apply(lambda x: x[:20] + "..." if len(x) > 20 else x)

        plot_save_label_counts(labels_count, os.path.join(REPORTS_DIR,
                                                          "data_plots",
                                                          exp_config.exp_name,
                                                          "test_sets",
                                                          f"test_{i}_titles_counts.png"))

    if exp_config.log_wandb:
        log_parameters(ds, exp_config.exp_name)

    print(f"Train set pickled to {LegalDataset.train_path}!")
    print(f"Validation set pickled to {LegalDataset.val_path}!")
    print(f"Test set list pickled to {LegalDataset.test_list_path}!")


if __name__ == "__main__":
    data_main(ExperimentConfig("we", "we", "we", seq_sampling_strategy="random"))
