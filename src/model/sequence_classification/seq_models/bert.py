import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast, BertForSequenceClassification

from src.model.sequence_classification.seq_models_interface import SeqClassification


# maybe consider composition rather than multiple inheritance
class FineTunedBert(BertForSequenceClassification, SeqClassification):

    def __init__(self, config, labels_weights: np.ndarray, tokenizer):

        BertForSequenceClassification.__init__(self, config)

        SeqClassification.__init__(
            self,
            tokenizer=tokenizer,
            optimizer=torch.optim.AdamW(list(self.parameters()), lr=2e-5)
        )

        self.labels_weights = torch.from_numpy(labels_weights).to(torch.float32)

    def tokenize(self, sample):

        output = self.tokenizer(', '.join(sample["input_title_sequence"]),
                                return_tensors='pt',
                                truncation=True)

        labels = torch.Tensor([self.config.label2id[sample["immediate_next_title"]]])

        return {'input_ids': output['input_ids'].squeeze(),
                'token_type_ids': output['token_type_ids'].squeeze(),
                'attention_mask': output['attention_mask'].squeeze(),
                'labels': labels.long()}

    def prepare_input(self, batch):
        input_dict = {}

        input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        token_type_ids = pad_sequence(batch["token_type_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(batch["attention_mask"], batch_first=True, padding_value=self.tokenizer.pad_token_id)

        input_dict["input_ids"] = input_ids.to(self.device)
        input_dict["token_type_ids"] = token_type_ids.to(self.device)
        input_dict["attention_mask"] = attention_mask.to(self.device)

        if "labels" in batch:
            input_dict["labels"] = batch["labels"].to(self.device)

        return input_dict

    def train_step(self, batch):

        output = self(**batch)

        loss = torch.nn.functional.cross_entropy(
            output.logits.view(-1, self.config.num_labels),
            batch["labels"].view(-1),
            weight=self.labels_weights.to(self.device)
        )

        return output.logits, loss

    @torch.no_grad()
    def valid_step(self, batch):

        output = self(**batch)

        val_loss = torch.nn.functional.cross_entropy(
            output.logits.view(-1, self.config.num_labels),
            batch["labels"].view(-1),
            weight=self.labels_weights.to(self.device)
        )

        predictions = output.logits.argmax(1)

        truth = batch['labels'].view(-1)

        acc = sum(
            x == y for x, y in zip(predictions, truth)
        )

        return acc, val_loss
