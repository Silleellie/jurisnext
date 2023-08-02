import os
import pickle
import random
from collections import namedtuple

import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

from src import DATA_DIR, RANDOM_STATE, SPLITS_DIR

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = "Legal dataset for BD Exam"

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "maybe_summarized_url?": "",
    "maybe-original_url?": "",
}

# Use a namedtuple just for better readability (access via dot notation)
CaseSequence = namedtuple("CaseSequence", ["title_sequence", "text_sequence", "rel_keywords_sequence"])


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class NewDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the
        #  dataset
        features = datasets.Features(
            {
                "case_id": datasets.Value("string"),
                "title_sequence": datasets.Sequence(feature=datasets.Value("string")),
                "text_sequence": datasets.Sequence(feature=datasets.Value("string")),
                "rel_keywords_sequence": datasets.Sequence(feature=datasets.Value("string"))
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the
        #  configuration

        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user
        # is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS It can
        # accept any type or nested list/dict and will give back the same structure with the url replaced with path
        # to local files. By default the archives will be extracted and a path to a cached folder where they are
        # extracted is returned instead of the archive
        # urls = _URLS[self.config.name]
        url = r"https://osf.io/download/9f8br/?view_only=eb94f7ba789241da87201437b25ef774"
        data_dir = dl_manager.download_and_extract(url)

        dataset: pd.DataFrame = pd.read_pickle(data_dir)

        # clean dataset by normalizing missing text paragraphs and removing useless columns
        dataset = self._clean_dataset(dataset)

        # convert dataset from df to dict of list, where key is case_id and val is a list with 2 elements:
        # 1: list of texts
        # 2: list of titles
        # Both lists are in 1:1 relationships
        dict_dataset = dataset.groupby("case_id")[['title', 'text', 'rel_keywords']].agg(list)
        dict_dataset = dict_dataset.T.to_dict('list')

        dict_dataset = {case_id: CaseSequence(val[0], val[1], val[2]) for case_id, val in dict_dataset.items()}

        # split cases in train, val, test (each split contains COMPLETE cases)
        train_path, validation_path, test_path = self._split_dataset(dict_dataset)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": train_path,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": validation_path,
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": test_path,
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(filepath, "rb") as f:
            split_data = pickle.load(f)

        for case_id, sequence_tuple in split_data.items():
            yield case_id, {
                "case_id": case_id,
                "title_sequence": sequence_tuple.title_sequence,
                "text_sequence": sequence_tuple.text_sequence,
                "rel_keywords_sequence": sequence_tuple.rel_keywords_sequence
            }

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

    def _split_dataset(self, dict_dataset: dict):
        unique_case_ids = list(dict_dataset.keys())

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

        train_set = {case_id: dict_dataset[case_id] for case_id in train_ids}
        validation_set = {case_id: dict_dataset[case_id] for case_id in val_ids}
        test_set = {case_id: dict_dataset[case_id] for case_id in test_ids}

        os.makedirs(SPLITS_DIR, exist_ok=True)

        # serialize splits
        train_path = os.path.join(SPLITS_DIR, "train.pkl")
        with open(train_path, "wb") as f:
            pickle.dump(train_set, f)

        validation_path = os.path.join(SPLITS_DIR, "validation.pkl")
        with open(validation_path, "wb") as f:
            pickle.dump(validation_set, f)

        test_path = os.path.join(SPLITS_DIR, "test.pkl")
        with open(test_path, "wb") as f:
            pickle.dump(test_set, f)

        # unique_label_train = set(train_set["title"])
        # unique_label_val = set(validation_set["title"])
        # unique_label_test = set(test_set["title"])
        #
        # val_label_unknown = unique_label_val.difference(unique_label_train)
        # test_label_unknown = unique_label_test.difference(unique_label_train)
        #
        # assert len(val_label_unknown) == 0, "Some labels appear in the val set but not in the train set!"
        # assert len(test_label_unknown) == 0, "Some labels appear in the test set but not in the train set!"

        return train_path, validation_path, test_path
