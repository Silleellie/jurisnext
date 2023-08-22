import os
import pickle
import random
from functools import cached_property
from pathlib import Path
from typing import Tuple, Dict

import datasets
import numpy as np
import pandas as pd
from datasets import Dataset, NamedSplit
from sklearn.model_selection import train_test_split

from src import RAW_DATA_DIR, RANDOM_STATE, INTERIM_DATA_DIR, PROCESSED_DATA_DIR


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

    return cleaned_dataset


class LegalDataset:
    cleaned_dataset_path: str = os.path.join(INTERIM_DATA_DIR, "cleaned_dataframe.pkl")
    train_path: str = os.path.join(PROCESSED_DATA_DIR, "train.pkl")
    val_path: str = os.path.join(PROCESSED_DATA_DIR, "validation.pkl")
    test_list_path: str = os.path.join(PROCESSED_DATA_DIR, "test_list.pkl")

    def __init__(self, n_test_sets: int = 10):

        self.train_df, self.val_df, self.test_df_list = self._generate_splits_and_sample(n_test_sets)

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

        return pd.unique(all_labels)

    def _generate_splits_and_sample(self, n_test_sets: int):

        print("Creating dataset splits...")

        # remove the splits to have a fresh start
        Path(self.train_path).unlink(missing_ok=True)
        Path(self.val_path).unlink(missing_ok=True)
        Path(self.test_list_path).unlink(missing_ok=True)

        cleaned_dataset: pd.DataFrame = pd.read_pickle(self.cleaned_dataset_path)
        train_dataset, val_dataset, test_dataset = self._split_dataset(cleaned_dataset)

        train_dataset = self._group_dataset(train_dataset, to_sample=False)
        val_dataset = self._group_dataset(val_dataset, to_sample=True)
        test_dataset = [self._group_dataset(test_dataset, to_sample=True) for _ in range(n_test_sets)]

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
            random_state=RANDOM_STATE,
            shuffle=True
        )

        train_ids, val_ids = train_test_split(
            train_ids,
            test_size=0.1,
            random_state=RANDOM_STATE,
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

        return grouped_split

    @staticmethod
    def perform_sampling(batch):

        # a sequence has at least 1 data point, but it can have more depending on the length of the sequence
        # We must ensure that at least an element can be used as test set
        # in the "sliding_training_size" is included the target item
        sliding_size = random.randint(1, len(batch["text_sequence"]) - 1)

        # TO DO: consider starting always from the initial paragraph,
        # rather than varying the starting point of the seq
        start_index = random.randint(0, len(batch["text_sequence"]) - sliding_size - 1)
        end_index = start_index + sliding_size

        return {
            "case_id": batch["case_id"],
            "input_text_sequence": batch["text_sequence"][start_index:end_index],
            "input_title_sequence": batch["title_sequence"][start_index:end_index],
            "input_keywords_sequence": batch["rel_keywords_sequence"][start_index:end_index],
            "immediate_next_text": batch["text_sequence"][end_index],
            "immediate_next_title": batch["title_sequence"][end_index],
            "immediate_next_rel_keywords": batch["rel_keywords_sequence"][end_index]
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
    def load_dataset(cls):
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

        return obj


if __name__ == "__main__":
    
    original_df_path = os.path.join(RAW_DATA_DIR, "pre-processed_representations.pkl")
    cleaned_df_output_path = os.path.join(INTERIM_DATA_DIR, "cleaned_dataframe.pkl")

    original_df: pd.DataFrame = pd.read_pickle(original_df_path)
    cleaned_df = clean_original_dataset(original_df)

    cleaned_df.to_pickle(cleaned_df_output_path)

    # the constructor will create and dump the splits
    ds = LegalDataset()
    dataset_dict = ds.get_hf_datasets()

    print(dataset_dict["train"])
    print(dataset_dict["validation"])
    print(dataset_dict["test"])

    # train and val will be merged
    dataset_dict_no_val = ds.get_hf_datasets(merge_train_val=True)

    print(dataset_dict["train"])
    print(dataset_dict["validation"])
    print(dataset_dict["test"])
