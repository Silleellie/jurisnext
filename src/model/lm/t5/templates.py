import random
from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np

PromptTarget = namedtuple("PromptTarget", ["input_prompt", "target_text"])


class Task(ABC):

    @abstractmethod
    def __call__(self, title_sequence, target_title):
        raise NotImplementedError


class NextTitlePrediction(Task):

    templates = {
        0: PromptTarget(
            input_prompt="Predict the next element of the following sequence:\n"
                         "{}",
            target_text="{}"
        ),
        1: PromptTarget(
            input_prompt="Previous titles:\n"
                         "{}",
            target_text="{}"
        )
    }

    def __call__(self, title_sequence, target_title):

        # random select of string separator for titles sequence and the prompt to use
        separator = ", " if random.getrandbits(1) else "; "
        input_prompt, target_text = random.choice(self.templates)  # random.choice applied to dict return a value

        list_to_text = separator.join(title_sequence)
        text = input_prompt.format(list_to_text)
        target = target_text.format(target_title)

        return text, target


class BoolNextTitlePrediction(Task):

    templates = {
        0: PromptTarget(
            input_prompt="Given the following title sequences:\n"
                         "{}\n\n"
                         "Is this the next title? {}",
            target_text="{}"
        ),
        1: PromptTarget(
            input_prompt="Previous titles:\n"
                         "{}\n\n"
                         "Next title: {}",
            target_text="{}"
        )
    }

    def __init__(self, all_titles: np.ndarray):
        self.all_titles = all_titles

    def __call__(self, title_sequence, target_title):
        # random select of string separator for titles sequence and the prompt to use
        separator = ", " if random.getrandbits(1) else "; "
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
