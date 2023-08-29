import gc
import os
import pickle
from typing import Dict, Union, List, Any, Optional

import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig

from src import ExperimentConfig
from src.data.legal_dataset import LegalDataset
from src.model.clustering import KMeansAlg, ClusterLabelMapper
from src.model.next_title_prediction.ntp_models.multimodal.encoders import CNNEncoder, LSTMEncoder
from src.model.next_title_prediction.ntp_models_abtract import NTPModel, NTPConfig
from src.model.next_title_prediction.ntp_trainer import NTPTrainer
from src.model.sentence_encoders import SentenceTransformerEncoder
from src.utils import seed_everything


class MultimodalFusionConfig(PretrainedConfig, NTPConfig):

    def __init__(
            self,
            image_encoder_params: dict = None,
            text_encoder_params: dict = None,
            max_seq_len: int = 100,
            labels_weights: list = None,
            device: str = 'cpu',
            **kwargs
    ):

        PretrainedConfig.__init__(self, **kwargs)
        NTPConfig.__init__(self, device)

        self.image_encoder_params = image_encoder_params
        self.text_encoder_params = text_encoder_params
        self.max_seq_len = max_seq_len

        self.labels_weights = labels_weights
        if self.labels_weights is not None:
            self.labels_weights = torch.from_numpy(np.array(labels_weights)).float().to(device)

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: Union[str, os.PathLike],
                        cache_dir: Optional[Union[str, os.PathLike]] = None,
                        force_download: bool = False,
                        local_files_only: bool = False,
                        token: Optional[Union[str, bool]] = None,
                        revision: str = "main",
                        **kwargs):

        inst, unused_kwargs = super().from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs
        )

        # when loaded from path, labels weights is transformed already into a tensor by init,
        # that's why we need this further check
        if isinstance(inst.labels_weights, list):
            inst.labels_weights = torch.from_numpy(np.array(inst.labels_weights)).float().to(inst.device)

        return inst, unused_kwargs

    def save_pretrained(self,
                        save_directory: Union[str, os.PathLike],
                        push_to_hub: bool = False,
                        **kwargs):

        self.labels_weights = self.labels_weights.tolist()

        super().save_pretrained(save_directory=save_directory,
                                push_to_hub=push_to_hub,
                                **kwargs)

        self.labels_weights = torch.from_numpy(np.array(self.labels_weights)).float().to(self.device)

    # to make __repr__ work we need to convert the tensor to a json serializable format
    def to_dict(self) -> Dict[str, Any]:
        super_dict = super().to_dict()

        if isinstance(super_dict["labels_weights"], torch.Tensor):
            super_dict["labels_weights"] = super_dict["labels_weights"].tolist()

        return super_dict


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

        super().__init__(config=config)

        self.head_module = torch.nn.Linear(self.output_dim, len(self.config.label2id))
        self.parameters_to_update.extend(self.head_module.parameters())

    def forward(self, x: Dict[str, Union[torch.Tensor, List[str]]]) -> torch.Tensor:

        multimodal_features = MultimodalFusion.forward(self, x)
        output = self.head_module(multimodal_features)

        return output


class NTPMultimodalFusion(NTPModel):

    model_class = MultimodalFusionForSequenceClassification

    def __init__(self, model: MultimodalFusionForSequenceClassification, cluster_label_mapper=None):

        super().__init__(
            model=model,
            tokenizer=AutoTokenizer.from_pretrained(model.config.text_encoder_params["model_name"]),
            cluster_label_mapper=cluster_label_mapper),

    def get_suggested_optimizer(self):
        return torch.optim.AdamW(self.model.parameters_to_update, lr=2e-5)

    def save(self, save_path):

        self.model.save_pretrained(save_path)

        if self.cluster_label_mapper is not None:
            with open(os.path.join(save_path, 'cluster_label_mapper.pkl'), "wb") as f:
                pickle.dump(self.cluster_label_mapper, f)

    @classmethod
    def load(cls, save_path):

        model = cls.model_class.from_pretrained(
            pretrained_model_name_or_path=save_path
        )

        cluster_label_mapper_path = os.path.join(save_path, 'cluster_label_mapper.pkl')

        cluster_label_mapper = None
        if os.path.isfile(cluster_label_mapper_path):
            with open(cluster_label_mapper_path, "rb") as f:
                cluster_label_mapper = pickle.load(f)

        new_inst = cls(
            model=model,
            cluster_label_mapper=cluster_label_mapper
        )

        return new_inst

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
            labels_for_cluster = self.cluster_label_mapper.get_labels_from_cluster(immediate_next_cluster)
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

        input_dict["image"] = batch["image"].to(self.model.device).float()
        input_dict["text"] = {}

        input_dict["text"]["input_ids"] = pad_sequence(batch["input_ids"], batch_first=True,
                                                       padding_value=self.tokenizer.pad_token_id).to(self.model.device)
        input_dict["text"]["token_type_ids"] = pad_sequence(batch["token_type_ids"], batch_first=True,
                                                            padding_value=self.tokenizer.pad_token_id).to(self.model.device)
        input_dict["text"]["attention_mask"] = pad_sequence(batch["attention_mask"], batch_first=True,
                                                            padding_value=self.tokenizer.pad_token_id).to(self.model.device)

        if "labels" in batch:
            input_dict["labels"] = batch["labels"].to(self.model.device).flatten()

        return input_dict

    def train_step(self, batch):
        output = self(batch)
        truth = batch["labels"]

        loss = torch.nn.functional.cross_entropy(
            output,
            truth,
            weight=self.config.labels_weights
        )

        return loss

    @torch.no_grad()
    def valid_step(self, batch):
        output = self(batch)
        truth = batch["labels"]

        predictions = output.argmax(1)

        val_loss = torch.nn.functional.cross_entropy(
            output,
            truth,
            weight=self.config.labels_weights
        )

        predictions = [self.config.id2label[x.cpu().item()] for x in predictions]
        truth = [self.config.id2label[x.cpu().item()] for x in truth]

        return predictions, truth, val_loss

def multimodal_main():

    n_epochs = ExperimentConfig.epochs
    batch_size = ExperimentConfig.batch_size
    eval_batch_size = ExperimentConfig.eval_batch_size
    device = ExperimentConfig.device
    use_cluster_alg = ExperimentConfig.use_cluster_alg
    random_state = ExperimentConfig.random_state

    seed_everything(random_state)

    ds = LegalDataset.load_dataset()
    dataset = ds.get_hf_datasets()
    all_unique_labels = ds.all_unique_labels

    cluster_label = None
    if use_cluster_alg:
        clus_alg = KMeansAlg(
            n_clusters=50,
            random_state=random_state,
            init="k-means++",
            n_init="auto"
        )

        sent_encoder = SentenceTransformerEncoder(
            device=device,
        )

        cluster_label = ClusterLabelMapper(sent_encoder, clus_alg)

    train = dataset["train"]
    val = dataset["validation"]
    test_list = dataset["test"]

    all_train_labels_occurrences = [y for x in train for y in x['title_sequence']]
    # "smoothing" so that a weight can be calculated for labels which do not appear in the
    # train set
    all_train_labels_occurrences.extend(all_unique_labels)

    labels_weights = compute_class_weight(class_weight='balanced',
                                          classes=all_unique_labels,
                                          y=all_train_labels_occurrences)

    model = MultimodalFusionForSequenceClassification(
        MultimodalFusionConfig(
            image_encoder_params={
                "input_dims": [1, 32, 64, 128, 64, 10],
                "output_dims": [32, 64, 128, 64, 10, 5],
                "kernel_sizes": [7, 5, 5, 5, 5, 1]
            },
            text_encoder_params={
                "model_name": "nlpaueb/legal-bert-base-uncased",
                "model_hidden_states_num": 4,
                "hidden_size": 256,
                "directions_fusion_strat": "mean"
            },
            max_seq_len=100,
            label2id={x: i for i, x in enumerate(all_unique_labels)},
            id2label={i: x for i, x in enumerate(all_unique_labels)},
            labels_weights=list(labels_weights),
            device='cuda:0'
        ),
    )

    model_ntp = NTPMultimodalFusion(
        model=model,
        cluster_label_mapper=cluster_label,
    )

    trainer = NTPTrainer(
        ntp_model=model_ntp,
        n_epochs=n_epochs,
        batch_size=batch_size,
        all_labels=all_unique_labels,
        eval_batch_size=eval_batch_size,
        output_name=f"MultimodalFusion_{n_epochs}"
    )

    trainer.train(train, val)

    gc.collect()
    torch.cuda.empty_cache()

    print("EVALUATION")
    trainer.ntp_model = NTPMultimodalFusion.load(trainer.output_path)

    acc = []
    for test in test_list:
        acc.append(trainer.evaluate(test))
    print(acc)
