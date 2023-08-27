from typing import Dict, Union, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig

from src.model.next_title_prediction.ntp_models.multimodal.encoders import CNNEncoder, LSTMEncoder
from src.model.next_title_prediction.ntp_models_interface import NextTitlePredictor


class MultimodalFusionConfig(PretrainedConfig):

    def __init__(
            self,
            image_encoder_params: dict = None,
            text_encoder_params: dict = None,
            max_seq_len: int = 100,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.image_encoder_params = image_encoder_params
        self.text_encoder_params = text_encoder_params
        self.max_seq_len = max_seq_len


class MultimodalFusion(PreTrainedModel):

    config_class = MultimodalFusionConfig

    def __init__(self, config: MultimodalFusionConfig):

        super().__init__(config=config)

        self.image_encoder = CNNEncoder(**self.config.image_encoder_params)
        self.text_encoder = LSTMEncoder(**self.config.text_encoder_params)

        ## COMPUTE EXPECTED OUTPUT DIMENSION FOR IMAGE ENCODER ##

        # encoded image repr will have a number of rows equivalent to the max sequence length specified in input,
        # number of columns equal to the number of labels and 1 channel ("gray image")
        h, w, c = self.config.max_seq_len, len(self.config.label2id), 1

        kernel_size = 2
        dilation = 1
        stride = 2

        for c_in, c_out in zip(self.config.image_encoder_params["input_dims"], self.config.image_encoder_params["output_dims"]):

            # formula to compute expected shape after MaxPool2D layer
            # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
            # N.B. Conv2D layers keeps same output dimension as input because of padding
            h = ((h - dilation * (kernel_size - 1) - 1) // stride) + 1
            w = ((w - dilation * (kernel_size - 1) - 1) // stride) + 1

            c = c_out

        image_output_dim = h * w * c

        ## COMPUTE EXPECTED OUTPUT DIMENSION FOR TEXT ENCODER ##

        text_output_dim = self.text_encoder.expected_output_size

        self.output_dim = image_output_dim + text_output_dim

        # done in this way so that the huggingface model parameters in the text encoder are not added
        parameters_to_update = []
        parameters_to_update.extend(self.text_encoder.lstm.parameters())
        parameters_to_update.extend(self.image_encoder.parameters())

        self.parameters_to_update = parameters_to_update

    def forward(self, x: Dict[str, Union[torch.Tensor, List[str]]]) -> torch.Tensor:

        assert 'text' in x, "'text' representation is missing in input dict!"
        assert 'image' in x, "'image' representation is missing in input dict!"

        image_output = self.image_encoder(x['image'])
        text_output = self.text_encoder(x['text'])

        multimodal_features = torch.concat([image_output, text_output], dim=1)

        return multimodal_features


class MultimodalFusionForSequenceClassification(MultimodalFusion):

    def __init__(self, config: MultimodalFusionConfig):

        MultimodalFusion.__init__(self, config=config)

        self.head_module = torch.nn.Linear(self.output_dim, len(self.config.label2id))
        self.parameters_to_update.extend(self.head_module.parameters())

    def forward(self, x: Dict[str, Union[torch.Tensor, List[str]]]) -> torch.Tensor:

        multimodal_features = MultimodalFusion.forward(self, x)
        output = self.head_module(multimodal_features)

        return output


class NextTitleMultimodalFusion(NextTitlePredictor):

    model_class = MultimodalFusionForSequenceClassification

    def __init__(self, model: MultimodalFusionForSequenceClassification, labels_weights, cluster_label_mapper=None, device: str = "cuda:0"):

        self.labels_weights = torch.from_numpy(labels_weights).to(torch.float32).to(device)

        NextTitlePredictor.__init__(
            self,
            model=model,
            tokenizer=AutoTokenizer.from_pretrained(model.config.text_encoder_params["model_name"]),
            optimizer=torch.optim.AdamW(model.parameters_to_update, lr=2e-5),
            cluster_label_mapper=cluster_label_mapper,
            device=device
        )

    def tokenize(self, sample):
        """
        Note: the tokenize function also creates the encoded image representation in this case differently from
        other sequence classification models
        """

        input_dict = {}

        # rows represent different sequences length (e.g. only first label or from first to fifth label)
        # columns represent all available labels
        # if a label appears in a sequence, the corresponding cell value is increased (+1)
        image_repr = []
        last_repr = np.full(len(self.config.label2id), 0)
        title_str = ", ".join(sample['input_title_sequence'])

        for title in sample['input_title_sequence']:
            title_idx = self.config.label2id[title]
            last_repr[title_idx] += 1
            image_repr.append(last_repr)

        # if max length of the sequence which represents the image is different from the model max length
        # add rows full of zeros until the max length is reached
        if len(image_repr) != self.config.max_seq_len:
            image_repr.extend(
                [torch.from_numpy(np.full(len(self.config.label2id), 0)) for _ in
                 range(self.config.max_seq_len - len(image_repr))])

        if self.cluster_label_mapper is not None:
            immediate_next_cluster = self.cluster_label_mapper.get_clusters_from_labels(sample["immediate_next_title"])
            title_str = title_str + f"\nNext title cluster: {immediate_next_cluster}"

            # to add cluster label information, an additional row is added to the image
            # this row will have all cells that represent labels that are part of the cluster with value 1 and
            # all other cells with value 0
            labels_for_cluster = self.cluster_label_mapper.get_labels_from_clusters(immediate_next_cluster)
            cluster_repr_image_row = np.full(len(self.config.label2id), 0)

            for label_in_cluster in labels_for_cluster:
                cluster_repr_image_row[self.config.label2id[label_in_cluster]] = 1

            # !!! The final number of rows in this case will be max_seq_len + 1 when using cluster labels !!!
            image_repr.append(cluster_repr_image_row)

        # unsqueeze to add channel and convert to [0, 1] range
        max_image_repr_value = np.max(image_repr)
        image_repr = torch.from_numpy(np.vstack(image_repr)).unsqueeze(0).float().div(max_image_repr_value)
        tokenizer_output = self.tokenizer(title_str, truncation=True)

        input_dict["input_ids"] = tokenizer_output["input_ids"]
        input_dict["token_type_ids"] = tokenizer_output["token_type_ids"]
        input_dict["attention_mask"] = tokenizer_output["attention_mask"]

        input_dict['image'] = image_repr
        input_dict['labels'] = [self.config.label2id[sample['immediate_next_title']]]

        return input_dict

    def prepare_input(self, batch):
        input_dict = {}

        input_dict["image"] = batch["image"].to(self.device).float()
        input_dict["text"] = {}

        input_dict["text"]["input_ids"] = pad_sequence(batch["input_ids"], batch_first=True,
                                                       padding_value=self.tokenizer.pad_token_id).to(self.device)
        input_dict["text"]["token_type_ids"] = pad_sequence(batch["token_type_ids"], batch_first=True,
                                                            padding_value=self.tokenizer.pad_token_id).to(self.device)
        input_dict["text"]["attention_mask"] = pad_sequence(batch["attention_mask"], batch_first=True,
                                                            padding_value=self.tokenizer.pad_token_id).to(self.device)

        if "labels" in batch:
            input_dict["labels"] = batch["labels"].to(self.device).flatten()

        return input_dict


    def train_step(self, batch):
        output = self(batch)
        truth = batch["labels"]

        loss = torch.nn.functional.cross_entropy(
            output,
            truth,
            weight=self.labels_weights
        )

        return output, loss

    @torch.no_grad()
    def valid_step(self, batch):
        output = self(batch)
        truth: torch.Tensor = batch["labels"]

        predictions = output.argmax(1)

        val_loss = torch.nn.functional.cross_entropy(
            output,
            truth,
            weight=self.labels_weights
        )

        acc = (predictions == truth).sum()

        return acc.item(), val_loss
