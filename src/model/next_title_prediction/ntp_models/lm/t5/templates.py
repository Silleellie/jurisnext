from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List

import numpy as np

PromptTarget = namedtuple("PromptTarget", ["input_prompt", "target_text"])


class Task(ABC):

    @abstractmethod
    def __call__(self, title_sequence, target_title, **kwargs):
        raise NotImplementedError

    def to_json(self):
        return repr(self), self.__dict__

    def __repr__(self):
        return self.__class__.__name__

    @classmethod
    def from_eval(cls, task_classname: str, task_parameters: dict) -> Task:
        eval_class = eval(task_classname)
        evaluated_task = eval_class(**task_parameters)

        return evaluated_task


class DirectNTP(Task):
    templates = {
        0: PromptTarget(
            input_prompt="DirectNTP:\n\n"
                         "Predict the next element of the following sequence ->\n"
                         "{}",
            target_text="{}"
        ),
        1: PromptTarget(
            input_prompt="DirectNTP:\n\n"
                         "Previous titles ->\n"
                         "{}\n"
                         "Next title is ",
            target_text="{}"
        )
    }

    def __call__(self, title_sequence, target_title, **kwargs):
        # random select of string separator for titles sequence and the prompt to use
        separator = " , " if random.getrandbits(1) else " ; "
        input_prompt, target_text = random.choice(self.templates)  # random.choice applied to dict return a value

        list_to_text = separator.join(title_sequence)
        text = input_prompt.format(list_to_text)
        target = target_text.format(target_title)

        return text, target


class DirectNTPSideInfo(Task):
    templates = {
        0: PromptTarget(
            input_prompt="DirectNTPSideInfo:\n\n"
                         "Predict the next element of the following sequence ->\n"
                         "{}\n"
                         "Relevant keywords for each element of the sequence are ->\n"
                         "{}",
            target_text="{}"
        ),
        1: PromptTarget(
            input_prompt="DirectNTPSideInfo:\n\n"
                         "Previous titles ->\n"
                         "{}\n"
                         "Context ->\n"
                         "{}"
                         "Next title is ",
            target_text="{}"
        )
    }

    def __init__(self, relevant_keywords_col: List[List[str]] = None, minimum_occ_number: int = None):

        self.unique_relevant_keywords = None
        self.minimum_occ_number = minimum_occ_number
        self.relevant_keywords_col = relevant_keywords_col

        if self.minimum_occ_number is not None and self.relevant_keywords_col is not None:

            # iterate over all words in the "rel_keywords" column
            iter_rel = lambda: (relevant_keywords.split(', ')
                                for relevant_keywords_seq in relevant_keywords_col
                                for relevant_keywords in relevant_keywords_seq)

            # need maximum len for dynamic collection of elements
            keywords_len = max(len(keyword) for keywords in iter_rel() for keyword in keywords)
            keywords = np.fromiter(itertools.chain.from_iterable(iter_rel()), dtype=f"U{keywords_len}")

            # find keywords which appear less than "minimum_occ_number" times and remove them
            unique_relevant_keywords, count_unique_relevant_keywords = np.unique(keywords, return_counts=True)
            mask_keywords_to_remove = count_unique_relevant_keywords >= minimum_occ_number
            self.unique_relevant_keywords = set(unique_relevant_keywords[mask_keywords_to_remove])

    def to_json(self):
        return repr(self), {"relevant_keywords_col": self.relevant_keywords_col,
                            "minimum_occ_number": self.minimum_occ_number}

    def __call__(self, title_sequence, target_title, **kwargs):
        assert "rel_keywords_seq" in kwargs, "rel_keywords should be set to use this template!"
        rel_keywords_seq = kwargs.pop("rel_keywords_seq")

        # since using all keywords of all elements of the sequence is too much,
        # we choose at random one for each element of the sequence

        reduced_rel_keywords = []

        if self.minimum_occ_number is not None and self.relevant_keywords_col is not None:

            for rel_keywords in rel_keywords_seq:

                keywords_seq = set(rel_keywords.split(", "))
                relevant_keywords_seq = keywords_seq.intersection(self.unique_relevant_keywords)

                # if there are no relevant keywords after removing the less occurring ones, choose one at random
                # among the less occurring ones. Otherwise, select one out of the most occurring ones
                if len(relevant_keywords_seq) == 0:
                    reduced_rel_keywords.append(random.sample(keywords_seq, 1)[0])
                else:
                    reduced_rel_keywords.append(random.sample(relevant_keywords_seq, 1)[0])

        else:

            # if no minimum_occ_number of relevant_keywords_col has been set, then select a random keyword from
            # all the ones associated with the title
            for rel_keywords in rel_keywords_seq:
                keywords_seq = set(rel_keywords.split(", "))
                reduced_rel_keywords.append(random.sample(keywords_seq, 1)[0])

        # random select of string separator for titles sequence and the prompt to use
        separator = " , " if random.getrandbits(1) else " ; "
        input_prompt, target_text = random.choice(self.templates)  # random.choice applied to dict return a value

        list_to_text = separator.join(title_sequence)
        list_to_rel_keywords = separator.join(reduced_rel_keywords)

        text = input_prompt.format(list_to_text, list_to_rel_keywords)
        target = target_text.format(target_title)

        return text, target


class BoolNTP(Task):
    templates = {
        0: PromptTarget(
            input_prompt="BoolNTP:\n\n"
                         "Given the following title sequences ->\n"
                         "{}\n\n"
                         "Is this the next title (answer yes/no)? {} ",
            target_text="{}"
        ),
        1: PromptTarget(
            input_prompt="BoolNTP:\n\n"
                         "Previous titles ->\n"
                         "{}\n\n"
                         "Answer yes/no if this is the next title -> {}",
            target_text="{}"
        )
    }

    def __init__(self, all_titles: List[str]):
        self.all_titles = np.array(all_titles)

    def to_json(self):
        return repr(self), {"all_titles": list(self.all_titles)}

    def __call__(self, title_sequence, target_title, **kwargs):
        # random select of string separator for titles sequence and the prompt to use
        separator = " , " if random.getrandbits(1) else " ; "
        input_prompt, target_text = random.choice(self.templates)  # random.choice applied to dict return a value

        # if randomly true, the next title is the correct one,
        # otherwise it is not
        if random.getrandbits(1):
            target_title = np.random.choice(self.all_titles[self.all_titles != target_title])
            target_text_label = "no"
        else:
            target_text_label = "yes"

        list_to_text = separator.join(title_sequence)
        text = input_prompt.format(list_to_text, target_title)
        target = target_text.format(target_text_label)

        return text, target


class ClusteredNTP(Task):
    templates = {
        0: PromptTarget(
            input_prompt="ClusteredNTP:\n\n"
                         "The title sequence is the following ->\n"
                         "{}\n\n"
                         "The next title is in cluster {}, what is the next element of the sequence?\n"
                         "Choose one among the following options ->\n"
                         "{}",  # bullet list
            target_text="{}"
        ),
        1: PromptTarget(
            input_prompt="ClusteredNTP:\n\n"
                         "Previous titles ->\n"
                         "{}\n"
                         "Next title cluster ->\n"
                         "{}\n\n"
                         "Chose the next title from the followings ->\n"
                         "{}",
            target_text="{}"
        )
    }

    def __call__(self, title_sequence, target_title, **kwargs):
        assert "cluster_mapper" in kwargs, "cluster_mapper should be set to use this template!"
        cluster_mapper = kwargs.pop("cluster_mapper")

        next_cluster = cluster_mapper.get_clusters_from_labels(target_title).item()
        next_possible_titles = cluster_mapper.get_labels_from_cluster(next_cluster)

        # random select of string separator for titles sequence and the prompt to use
        separator = " , " if random.getrandbits(1) else " ; "
        bullet_notation = " - " if random.getrandbits(1) else " * "
        input_prompt, target_text = random.choice(self.templates)  # random.choice applied to dict return a value

        list_to_text = separator.join(title_sequence)
        possible_titles_to_bullet_list = (f"{bullet_notation} {{}}\n" * len(next_possible_titles)).format(
            *next_possible_titles)

        text = input_prompt.format(list_to_text, next_cluster, possible_titles_to_bullet_list)
        target = target_text.format(target_title)

        return text, target


class ClusteredNTPSideInfo(Task):
    templates = {
        0: PromptTarget(
            input_prompt="ClusteredNTPSideInfo:\n\n"
                         "The title sequence is the following ->\n"
                         "{}\n"
                         "Relevant keywords of the sequence are ->\n"
                         "{}\n"
                         "The next title is in cluster {}, what is the next element of the sequence?\n"
                         "Choose one among the following options ->\n"
                         "{}",  # bullet list
            target_text="{}"
        ),
        1: PromptTarget(
            input_prompt="ClusteredNTPSideInfo:\n\n"
                         "Previous titles ->\n"
                         "{}\n"
                         "Relevant keywords ->\n"
                         "{}\n"
                         "Next title cluster ->\n"
                         "{}\n"
                         "Chose one of the followings ->\n"
                         "{}",
            target_text="{}"
        )
    }

    def __call__(self, title_sequence, target_title, **kwargs):
        assert "cluster_mapper" in kwargs, "cluster_mapper should be set to use this template!"
        assert "rel_keywords_seq" in kwargs, "rel_keywords should be set to use this template!"
        cluster_mapper = kwargs.pop("cluster_mapper")
        rel_keywords_seq = kwargs.pop("rel_keywords_seq")

        # since using all keywords of all elements of the sequence is too much,
        # we choose at random one for each element of the sequence
        reduced_rel_keywords = [random.choice(rel_keywords.split(", ")) for rel_keywords in rel_keywords_seq]

        next_cluster = cluster_mapper.get_clusters_from_labels(target_title).item()
        next_possible_titles = cluster_mapper.get_labels_from_cluster(next_cluster)

        # random select of string separator for titles sequence and the prompt to use
        separator = " , " if random.getrandbits(1) else " ; "
        bullet_notation = " - " if random.getrandbits(1) else " * "
        input_prompt, target_text = random.choice(self.templates)  # random.choice applied to dict return a value

        list_to_text = separator.join(title_sequence)
        list_to_rel_keywords = separator.join(reduced_rel_keywords)
        possible_titles_to_bullet_list = (f"{bullet_notation} {{}}\n" * len(next_possible_titles)).format(
            *next_possible_titles)

        text = input_prompt.format(list_to_text, list_to_rel_keywords, next_cluster, possible_titles_to_bullet_list)
        target = target_text.format(target_title)

        return text, target
