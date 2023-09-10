from __future__ import annotations
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

    def __call__(self, title_sequence, target_title, **kwargs):

        assert "rel_keywords_seq" in kwargs, "rel_keywords should be set to use this template!"
        rel_keywords_seq = kwargs.pop("rel_keywords_seq")

        # since using all keywords of all elements of the sequence is too much,
        # we choose at random one for each element of the sequence
        reduced_rel_keywords = [random.choice(rel_keywords.split(", ")) for rel_keywords in rel_keywords_seq]

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
        possible_titles_to_bullet_list = (f"{bullet_notation} {{}}\n" * len(next_possible_titles)).format(*next_possible_titles)

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
        possible_titles_to_bullet_list = (f"{bullet_notation} {{}}\n" * len(next_possible_titles)).format(*next_possible_titles)

        text = input_prompt.format(list_to_text, list_to_rel_keywords, next_cluster, possible_titles_to_bullet_list)
        target = target_text.format(target_title)

        return text, target
