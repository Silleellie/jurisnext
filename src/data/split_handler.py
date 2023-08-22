import os
import pickle
import random

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

from src import RAW_DATA_DIR, RANDOM_STATE, SPLITS_DIR


class SplitHandler:

    def __init__(self):

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_dataset(self):

        train_path = os.path.join(SPLITS_DIR, "train.pkl")
        validation_path = os.path.join(SPLITS_DIR, "validation.pkl")
        test_path = os.path.join(SPLITS_DIR, "test.pkl")

        if not os.path.isfile(train_path) or not os.path.isfile(validation_path) or not os.path.isfile(test_path):

            with open(os.path.join(RAW_DATA_DIR, "pre-processed_representations.pkl"), "rb") as f:
                dataset: pd.DataFrame = pickle.load(f)

            dataset = self._clean_dataset(dataset)
            train_dataset, val_dataset, test_dataset = self._split_dataset(dataset)

            train_dataset = self._group_dataset(train_dataset, False)
            val_dataset = self._group_dataset(val_dataset, True)
            test_dataset = self._group_dataset(test_dataset, True)

            with open(train_path, "wb") as f:
                pickle.dump(train_dataset, f)

            with open(validation_path, "wb") as f:
                pickle.dump(val_dataset, f)

            with open(test_path, "wb") as f:
                pickle.dump(test_dataset, f)

        else:

            with open(train_path, "rb") as f:
                train_dataset: pd.DataFrame = pickle.load(f)

            with open(validation_path, "rb") as f:
                val_dataset: pd.DataFrame = pickle.load(f)

            with open(test_path, "rb") as f:
                test_dataset: pd.DataFrame = pickle.load(f)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        return self

    def _clean_dataset(self, dataset: pd.DataFrame):
        dataset["concept:name:paragraphs"] = dataset["concept:name:paragraphs"].apply("\n".join).replace("",
                                                                                                         "!!No paragraph content!!")

        # some elements have [None] as concept:name:section, we substitute them with []
        dataset["concept:name:section"] = dataset["concept:name:section"].apply(lambda x: [] if all(el is None for el in x) else x)
        dataset["concept:name:section"] = dataset["concept:name:section"].apply(", ".join).replace("",
                                                                                                   "!!No paragraph content!!")

        dataset = dataset[["case:concept:name", "concept:name", "concept:name:paragraphs", "concept:name:section"]]

        dataset = dataset.rename(columns={"case:concept:name": "case_id",
                                          "concept:name": "title",
                                          "concept:name:paragraphs": "text",
                                          "concept:name:section": "rel_keywords"})

        return dataset

    def _split_dataset(self, dataset: pd.DataFrame):

        unique_case_ids = list(pd.unique(dataset["case_id"]))

        train_val_ids, test_ids = train_test_split(
            unique_case_ids,
            test_size=0.2,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=0.1,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        train_set = dataset[dataset['case_id'].isin(train_ids)]
        val_set = dataset[dataset['case_id'].isin(val_ids)]
        test_set = dataset[dataset['case_id'].isin(test_ids)]

        return train_set, val_set, test_set

    def _group_dataset(self, dataset_split, to_sample: bool = False):

        grouped_split = dataset_split.groupby("case_id")[['title', 'text', 'rel_keywords']].agg(list)

        grouped_split = grouped_split.rename(columns={
            'title': 'title_sequence',
            'text': 'text_sequence',
            'rel_keywords': 'rel_keywords_sequence'
        })

        if to_sample:
            sampled_split = grouped_split.apply(self._perform_sampling, axis=1)
            grouped_split = pd.DataFrame.from_records(sampled_split)
            grouped_split['case_id'] = sampled_split.index

        return grouped_split

    def _perform_sampling(self, batch):

        # a sequence has at least 1 data point, but it can have more depending on the length of the sequence
        # We must ensure that at least an element can be used as test set
        # in the "sliding_training_size" is included the target item
        sliding_size = random.randint(1, len(batch["text_sequence"]) - 1)

        # TO DO: consider starting always from the initial paragraph,
        # rather than varying the starting point of the seq
        start_index = random.randint(0, len(batch["text_sequence"]) - sliding_size - 1)
        end_index = start_index + sliding_size

        return {
            "input_text_sequence": batch["text_sequence"][start_index:end_index],
            "input_title_sequence": batch["title_sequence"][start_index:end_index],
            "input_keywords_sequence": batch["rel_keywords_sequence"][start_index:end_index],
            "immediate_next_text": batch["text_sequence"][end_index],
            "immediate_next_title": batch["title_sequence"][end_index],
            "immediate_next_rel_keywords": batch["rel_keywords_sequence"][end_index]
        }

    def get_hf_splits(self):

        assert self.train_dataset is not None, "No train dataset loaded, call 'load_dataset' method first!"
        assert self.val_dataset is not None, "No validation dataset loaded, call 'load_dataset' method first!"
        assert self.test_dataset is not None, "No test dataset loaded, call 'load_dataset' method first!"

        return Dataset.from_pandas(self.train_dataset), \
            Dataset.from_pandas(self.val_dataset), \
            Dataset.from_pandas(self.test_dataset)


if __name__ == "__main__":

    ds = SplitHandler().load_dataset()
    train, val, test = ds.get_hf_splits()

    print(train)
    print(val)
    print(test)

