from abc import abstractmethod


# interface for all sequence classification models
class SeqClassification:

    def __init__(self, tokenizer, optimizer):
        self.tokenizer = tokenizer
        self.optimizer = optimizer

    @abstractmethod
    def tokenize(self, sample):

        raise NotImplementedError

    @abstractmethod
    def prepare_input(self, batch):

        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch):

        raise NotImplementedError

    @abstractmethod
    def valid_step(self, batch):

        raise NotImplementedError
