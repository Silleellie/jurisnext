from typing import Dict, Union, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig

from src import ExperimentConfig
from src.data.legal_dataset import LegalDataset
from src.model.next_title_prediction.ntp_models.custom_encoders.encoders import CNNEncoder, LSTMEncoder
from src.model.next_title_prediction.ntp_models_abtract import NTPModel, NTPConfig
from src.model.next_title_prediction.ntp_trainer import NTPTrainer


class MultimodalFusionConfig(PretrainedConfig, NTPConfig):

    def __init__(
            self,
            cnn_encoder_params: dict = None,
            lstm_encoder_params: dict = None,
            max_seq_len: int = 100,
            device: str = 'cpu',
            **kwargs
    ):

        PretrainedConfig.__init__(self, **kwargs)
        NTPConfig.__init__(self, device)

        self.cnn_encoder_params = cnn_encoder_params
        self.lstm_encoder_params = lstm_encoder_params
        self.max_seq_len = max_seq_len


class MultimodalFusion(PreTrainedModel):

    config_class = MultimodalFusionConfig

    def __init__(self, config: MultimodalFusionConfig):

        super().__init__(config=config)

        self.cnn_encoder = CNNEncoder(**self.config.cnn_encoder_params)
        self.lstm_encoder = LSTMEncoder(**self.config.lstm_encoder_params)

        ## COMPUTE EXPECTED OUTPUT DIMENSION FOR CNN ENCODER ##

        # encoded image repr will have a number of rows equivalent to the max sequence length specified in input,
        # number of columns equal to the number of labels and 1 channel ("gray image")
        h, w, c = self.config.max_seq_len, len(self.config.label2id), 1

        kernel_size = 2
        dilation = 1
        stride = 2

        for c_in, c_out in zip(self.config.cnn_encoder_params["input_dims"], self.config.cnn_encoder_params["output_dims"]):

            # formula to compute expected shape after MaxPool2D layer
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
            # N.B. Conv2D layers keeps same output dimension as input because of padding
            h = ((h - dilation * (kernel_size - 1) - 1) // stride) + 1
            w = ((w - dilation * (kernel_size - 1) - 1) // stride) + 1

            c = c_out

        image_output_dim = h * w * c

        ## COMPUTE EXPECTED OUTPUT DIMENSION FOR LSTM ENCODER ##

        text_output_dim = self.lstm_encoder.expected_output_size

        self.output_dim = image_output_dim + text_output_dim

        # done in this way so that the huggingface model parameters in the text encoder are not added
        parameters_to_update = []
        parameters_to_update.extend(self.lstm_encoder.parameters())
        parameters_to_update.extend(self.cnn_encoder.parameters())

        self.parameters_to_update = parameters_to_update

    def forward(self, x: Dict[str, Union[torch.Tensor, List[str]]]) -> torch.Tensor:

        assert 'text' in x, "'text' is missing in input dict!"
        assert 'image' in x, "'image' is missing in input dict!"

        image_output = self.cnn_encoder(x['image'])
        text_output = self.lstm_encoder(x['text'])

        concat_features = torch.concat([image_output, text_output], dim=1)

        return concat_features


class MultimodalFusionForSequenceClassification(MultimodalFusion):

    def __init__(self, config: MultimodalFusionConfig):

        super().__init__(config=config)

        self.head_module = torch.nn.Linear(self.output_dim, len(self.config.label2id))
        self.parameters_to_update.extend(self.head_module.parameters())

    def forward(self, x: Dict[str, Union[torch.Tensor, List[str]]]) -> torch.Tensor:

        multimodal_features = MultimodalFusion.forward(self, x)
        output = self.head_module(multimodal_features)

        return output


class NTPMultimodalFusion(NTPModel):

    model_class = MultimodalFusionForSequenceClassification

    def __init__(self, model: MultimodalFusionForSequenceClassification):

        super().__init__(
            model=model,
            tokenizer=AutoTokenizer.from_pretrained(model.config.lstm_encoder_params["model_name"])),

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
        title_str = ", ".join(sample['input_title_sequence'])

        tokenizer_output = self.tokenizer(title_str, truncation=True)

        input_dict["input_ids"] = tokenizer_output["input_ids"]
        input_dict["token_type_ids"] = tokenizer_output["token_type_ids"]
        input_dict["attention_mask"] = tokenizer_output["attention_mask"]

        input_dict['labels'] = [self.config.label2id[sample['immediate_next_title']]]
        input_dict['titles'] = sample['input_title_sequence']

        return input_dict

    def prepare_input(self, batch):
        input_dict = {}

        image_reprs = []
        for titles in batch['titles']:

            # rows represent different sequences length (e.g. only first label or from first to fifth label)
            # columns represent all available labels
            # if a label appears in a sequence, the corresponding cell value is increased (+1)
            image_repr = []
            last_repr = np.full(len(self.config.label2id), 0)

            for title in titles:
                title_idx = self.config.label2id[title]
                last_repr[title_idx] += 1
                image_repr.append(last_repr.copy())

            # unsqueeze to add channel and convert to [0, 1] range
            image_repr = np.vstack(image_repr)
            # if max length of the sequence which represents the image is different from the model max length
            # add rows full of zeros until the max length is reached
            image_repr = np.pad(np.vstack(image_repr), [(0, self.config.max_seq_len - len(image_repr)), (0, 0)])
            max_image_repr_value = np.max(image_repr)
            image_repr = torch.from_numpy(image_repr).unsqueeze(0).float().div(max_image_repr_value)
            image_reprs.append(image_repr)

        input_dict['image'] = torch.stack(image_reprs).to(self.model.device).float()

        input_dict["text"] = {}

        input_dict["text"]["input_ids"] = pad_sequence(batch["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.model.device)
        input_dict["text"]["token_type_ids"] = pad_sequence(batch["token_type_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.model.device)
        input_dict["text"]["attention_mask"] = pad_sequence(batch["attention_mask"], batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.model.device)

        if "labels" in batch:
            input_dict["labels"] = batch["labels"].to(self.model.device).flatten()

        return input_dict

    def train_step(self, batch):
        output = self(batch)
        truth = batch["labels"]

        loss = torch.nn.functional.cross_entropy(
            output,
            truth
        )

        return loss

    @torch.no_grad()
    def valid_step(self, batch):
        output = self(batch)
        truth = batch["labels"]

        predictions = output.argmax(1)

        val_loss = torch.nn.functional.cross_entropy(
            output,
            truth
        )

        predictions = [self.config.id2label[x.cpu().item()] for x in predictions]
        truth = [self.config.id2label[x.cpu().item()] for x in truth]

        return predictions, truth, val_loss


def fusion_main(exp_config: ExperimentConfig):

    freeze_emb_model = exp_config.freeze_emb_model
    n_epochs = exp_config.epochs
    batch_size = exp_config.train_batch_size
    eval_batch_size = exp_config.eval_batch_size
    device = exp_config.device

    ds = LegalDataset.load_dataset(exp_config)
    dataset = ds.get_hf_datasets()
    all_unique_labels = ds.all_unique_labels
    sampling_fn = ds.perform_sampling

    train = dataset["train"]
    val = dataset["validation"]

    model = MultimodalFusionForSequenceClassification(
        MultimodalFusionConfig(
            cnn_encoder_params={
                "input_dims": [1, 64, 128, 128, 64, 64],
                "output_dims": [64, 128, 128, 64, 64, 10],
                "kernel_sizes": [7, 5, 5, 5, 5, 1]
            },
            lstm_encoder_params={
                "model_name": "bert-base-uncased",
                "model_hidden_states_num": 4,
                "directions_fusion_strat": "concat",
                "freeze_embedding_model": freeze_emb_model
            },
            max_seq_len=100,
            label2id={x: i for i, x in enumerate(all_unique_labels)},
            id2label={i: x for i, x in enumerate(all_unique_labels)},
            device=device
        ),
    )

    model_ntp = NTPMultimodalFusion(
        model=model,
    )

    output_name = f"CNN_LSTM_{n_epochs}"
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
        train_sampling_fn=sampling_fn,
        monitor_strategy=exp_config.monitor_strategy
    )

    trainer.train(train, val)

    return trainer.output_name
