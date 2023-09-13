from abc import abstractmethod, ABC
from typing import Union, Tuple, List

import torch
import transformers.optimization
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer


class NTPConfig:

    def __init__(self, device: str = "cpu"):
        self.device = device


# interface for all sequence classification models
class NTPModel(ABC):

    model_class = None
    tokenizer_class = AutoTokenizer

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):

        self.model = model
        self.tokenizer = tokenizer

        self.model.to(self.model.config.device)

    # returns optimizer used in the default experiments
    @abstractmethod
    def get_suggested_optimizer(self) -> Union[torch.optim.Optimizer, transformers.optimization.Optimizer]:
        raise NotImplementedError

    # returns the tokenized version of inputs for the model + additional info needed
    @abstractmethod
    def tokenize(self, sample) -> dict:
        raise NotImplementedError

    # performs additional ops on the tokenized input batch (e.g. for t5, -100 for pad token in target_ids)
    @abstractmethod
    def prepare_input(self, batch) -> dict:
        raise NotImplementedError

    # returns loss
    @abstractmethod
    def train_step(self, batch) -> torch.Tensor:
        raise NotImplementedError

    # return predictions, truths as string labels and loss
    @abstractmethod
    def valid_step(self, batch) -> Tuple[List[str], List[str], torch.Tensor]:
        raise NotImplementedError

    @property
    def config(self):
        return self.model.config

    def train(self, mode: bool = True):
        return self.model.train(mode)

    def eval(self):
        return self.model.eval()

    @property
    def training(self):
        return self.model.training

    def save(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @classmethod
    def load(cls, save_path):

        model = cls.model_class.from_pretrained(
            pretrained_model_name_or_path=save_path
        )

        tokenizer = cls.tokenizer_class.from_pretrained(
            pretrained_model_name_or_path=save_path
        )

        new_inst = cls(
            model=model,
            tokenizer=tokenizer
        )

        return new_inst

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class NTPModelHF(NTPModel):

    config_class = NTPConfig

    def __init__(self,
                 pretrained_model_or_pth: str,
                 **config_kwargs):

        self.model_class.config_class = self.config_class
        model = self.model_class.from_pretrained(pretrained_model_or_pth, **config_kwargs)
        tokenizer = self.tokenizer_class.from_pretrained(pretrained_model_or_pth)

        super().__init__(model, tokenizer)

    @classmethod
    def load(cls, save_path):
        cls.model_class.config_class = cls.config_class

        return cls(
            pretrained_model_or_pth=save_path
        )
