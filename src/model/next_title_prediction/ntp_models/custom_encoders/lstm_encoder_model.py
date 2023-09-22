from typing import Dict, Union, List

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig

from src import ExperimentConfig
from src.data.legal_dataset import LegalDataset
from src.model.next_title_prediction.ntp_models.custom_encoders.encoders import LSTMEncoder
from src.model.next_title_prediction.ntp_models_abtract import NTPModel, NTPConfig
from src.model.next_title_prediction.ntp_trainer import NTPTrainer


class LSTMConfig(PretrainedConfig, NTPConfig):

    def __init__(
            self,
            lstm_encoder_params: dict = None,
            device: str = 'cpu',
            **kwargs
    ):

        PretrainedConfig.__init__(self, **kwargs)
        NTPConfig.__init__(self, device)

        self.lstm_encoder_params = lstm_encoder_params


class LSTMModel(PreTrainedModel):
    config_class = LSTMConfig

    def __init__(self, config: LSTMConfig):
        super().__init__(config=config)

        self.lstm_encoder = LSTMEncoder(**self.config.lstm_encoder_params)

        self.output_dim = self.lstm_encoder.expected_output_size
        self.parameters_to_update = []
        self.parameters_to_update.extend(filter(lambda p: p.requires_grad, self.lstm_encoder.parameters()))

    def forward(self, x: Dict[str, Union[torch.Tensor, List[str]]]) -> torch.Tensor:
        lstm_output = self.lstm_encoder(x)
        return lstm_output


class LSTMModelForSequenceClassification(LSTMModel):

    def __init__(self, config: LSTMConfig):
        super().__init__(config=config)

        self.head_module = torch.nn.Linear(self.output_dim, len(self.config.label2id))
        self.parameters_to_update.extend(self.head_module.parameters())

    def forward(self, x: Dict[str, Union[torch.Tensor, List[str]]]) -> torch.Tensor:
        lstm_features = LSTMModel.forward(self, x)
        output = self.head_module(lstm_features)

        return output


class NTPLSTMModel(NTPModel):
    model_class = LSTMModelForSequenceClassification

    def __init__(self, model: LSTMModelForSequenceClassification):
        super().__init__(
            model=model,
            tokenizer=AutoTokenizer.from_pretrained(model.config.lstm_encoder_params["model_name"]))

    def get_suggested_optimizer(self):
        return torch.optim.AdamW(self.model.parameters_to_update, lr=2e-5)

    def save(self, save_path):
        self.model.save_pretrained(save_path)

    @classmethod
    def load(cls, save_path):
        model = cls.model_class.from_pretrained(
            pretrained_model_name_or_path=save_path
        )

        new_inst = cls(
            model=model,
        )

        return new_inst

    def tokenize(self, sample):

        input_dict = {}

        tokenizer_output = self.tokenizer(', '.join(sample['input_title_sequence']), truncation=True)

        input_dict["input_ids"] = tokenizer_output["input_ids"]
        input_dict["token_type_ids"] = tokenizer_output["token_type_ids"]
        input_dict["attention_mask"] = tokenizer_output["attention_mask"]

        input_dict['labels'] = [self.config.label2id[sample['immediate_next_title']]]

        return input_dict

    def prepare_input(self, batch):
        input_dict = {"text": {}}

        input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        token_type_ids = pad_sequence(batch["token_type_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(batch["attention_mask"], batch_first=True, padding_value=self.tokenizer.pad_token_id)

        input_dict["text"]["input_ids"] = input_ids.to(self.model.device)
        input_dict["text"]["token_type_ids"] = token_type_ids.to(self.model.device)
        input_dict["text"]["attention_mask"] = attention_mask.to(self.model.device)

        if "labels" in batch:
            input_dict["labels"] = batch["labels"].to(self.model.device).flatten()

        return input_dict

    def train_step(self, batch):

        truth = batch.pop("labels")
        output = self(batch["text"])

        loss = torch.nn.functional.cross_entropy(
            output,
            truth
        )

        return loss

    @torch.no_grad()
    def valid_step(self, batch):

        truth = batch.pop("labels")
        output = self(batch["text"])

        predictions = output.argmax(1)

        val_loss = torch.nn.functional.cross_entropy(
            output,
            truth
        )

        predictions = [self.config.id2label[x.cpu().item()] for x in predictions]
        truth = [self.config.id2label[x.cpu().item()] for x in truth]

        return predictions, truth, val_loss


def lstm_model_main(exp_config: ExperimentConfig):

    freeze_emb_model = exp_config.freeze_emb_model
    n_epochs = exp_config.epochs
    batch_size = exp_config.train_batch_size
    eval_batch_size = exp_config.eval_batch_size
    device = exp_config.device
    random_seed = exp_config.random_seed

    ds = LegalDataset.load_dataset(exp_config)
    dataset = ds.get_hf_datasets()
    all_unique_labels = ds.all_unique_labels
    sampling_fn = ds.perform_sampling

    train = dataset["train"]
    val = dataset["validation"]

    model = LSTMModelForSequenceClassification(
        LSTMConfig(
            lstm_encoder_params={
                "model_name": "bert-base-uncased",
                "model_hidden_states_num": 4,
                "directions_fusion_strat": "concat",
                "freeze_embedding_model": freeze_emb_model
            },
            label2id={x: i for i, x in enumerate(all_unique_labels)},
            id2label={i: x for i, x in enumerate(all_unique_labels)},
            device=device
        ),
    )

    model_ntp = NTPLSTMModel(
        model=model,
    )

    output_name = f"LSTMModel_{n_epochs}"
    if exp_config.exp_name is not None:
        output_name = exp_config.exp_name

    trainer = NTPTrainer(
        ntp_model=model_ntp,
        n_epochs=n_epochs,
        batch_size=batch_size,
        all_labels=all_unique_labels,
        eval_batch_size=eval_batch_size,
        output_name=output_name,
        log_wandb=exp_config.log_wandb,
        random_seed=random_seed,
        train_sampling_fn=sampling_fn
    )

    trainer.train(train, val)

    return trainer.output_name


if __name__ == "__main__":
    lstm_model_main(ExperimentConfig("text", None, "we", t5_tasks=None, pipeline_phases=['train'], train_batch_size=16))
