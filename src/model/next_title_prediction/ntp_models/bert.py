from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification, BertConfig

from src import ExperimentConfig
from src.data.legal_dataset import LegalDataset
from src.model.clustering import ClusterLabelMapper, KMeansAlg
from src.model.next_title_prediction.ntp_models_abtract import NTPModelHF, NTPConfig, NTPModel
from src.model.next_title_prediction.ntp_trainer import NTPTrainer
from src.model.sentence_encoders import SentenceTransformerEncoder


class NTPBertConfig(BertConfig, NTPConfig):

    def __init__(self,
                 device: str = "cpu",
                 **kwargs):
        BertConfig.__init__(self, **kwargs)
        NTPConfig.__init__(self, device)


# maybe consider composition rather than multiple inheritance
class NTPBert(NTPModelHF):
    model_class = BertForSequenceClassification
    config_class = NTPBertConfig
    default_checkpoint = 'bert-base-uncased'

    def __init__(self,
                 pretrained_model_or_pth: str = default_checkpoint,
                 cluster_label_mapper: ClusterLabelMapper = None,
                 prediction_supporter: NTPModel = None,
                 **config_kwargs):

        super().__init__(
            pretrained_model_or_pth=pretrained_model_or_pth,
            cluster_label_mapper=cluster_label_mapper,
            prediction_supporter=prediction_supporter,
            **config_kwargs
        )

    def get_suggested_optimizer(self):
        return torch.optim.AdamW(list(self.model.parameters()), lr=2e-5)

    def tokenize(self, sample):

        next_title = sample["immediate_next_title"]

        if self.prediction_supporter is not None:
            tokenized_sample = self.prediction_supporter.tokenize(sample)

            tokenized_sample["input_ids"] = torch.LongTensor([tokenized_sample["input_ids"]]).to(
                self.prediction_supporter.model.device)
            tokenized_sample["token_type_ids"] = torch.LongTensor([tokenized_sample["token_type_ids"]]).to(
                self.prediction_supporter.model.device)
            tokenized_sample["attention_mask"] = torch.LongTensor([tokenized_sample["attention_mask"]]).to(
                self.prediction_supporter.model.device)
            tokenized_sample["labels"] = torch.LongTensor(tokenized_sample["labels"]).to(
                self.prediction_supporter.model.device).flatten()

            with torch.no_grad():
                output_supp = self.prediction_supporter.model(**tokenized_sample)

            predicted_cluster = output_supp.logits.argmax(dim=1).item()
            predicted_cluster_str = ClusterLabelMapper.template_label.format(str(predicted_cluster))

            output = self.tokenizer(', '.join(sample["input_title_sequence"]) +
                                    f"\n Next immediate cluster: {predicted_cluster_str}",
                                    truncation=True)

        else:

            output = self.tokenizer(', '.join(sample["input_title_sequence"]), truncation=True)

        if self.cluster_label_mapper is not None:
            next_title = ClusterLabelMapper.template_label.format(
                str(self.cluster_label_mapper.predict(sample["immediate_next_title"]))
            )

        labels = [self.config.label2id[next_title]]

        return {'input_ids': output['input_ids'],
                'token_type_ids': output['token_type_ids'],
                'attention_mask': output['attention_mask'],
                'labels': labels}

    def prepare_input(self, batch):
        input_dict = {}

        input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        token_type_ids = pad_sequence(batch["token_type_ids"], batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(batch["attention_mask"], batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)

        input_dict["input_ids"] = input_ids.to(self.model.device)
        input_dict["token_type_ids"] = token_type_ids.to(self.model.device)
        input_dict["attention_mask"] = attention_mask.to(self.model.device)

        if "labels" in batch:
            input_dict["labels"] = batch["labels"].to(self.model.device).flatten()

        return input_dict

    def train_step(self, batch):
        output = self(**batch)
        return output.loss

    @torch.no_grad()
    def valid_step(self, batch):
        output = self(**batch)
        truth = batch["labels"]
        # batch size (batch_size, num_labels) -> (batch_size, 1)
        predictions = output.logits.argmax(dim=1)

        predictions = [self.config.id2label[x.cpu().item()] for x in predictions]
        truth = [self.config.id2label[x.cpu().item()] for x in truth]

        return predictions, truth, output.loss


def bert_main(exp_config: ExperimentConfig):

    n_epochs = exp_config.epochs
    batch_size = exp_config.train_batch_size
    eval_batch_size = exp_config.eval_batch_size
    device = exp_config.device

    checkpoint = "bert-base-uncased"
    if exp_config.checkpoint is not None:
        checkpoint = exp_config.checkpoint

    ds = LegalDataset.load_dataset(exp_config)
    dataset = ds.get_hf_datasets()
    all_unique_labels = ds.all_unique_labels
    sampling_fn = ds.perform_sampling

    train = dataset["train"]
    val = dataset["validation"]

    if exp_config.k_clusters is None:

        labels_to_predict = all_unique_labels
        cluster_label = None

    else:

        labels_to_predict = [ClusterLabelMapper.template_label.format(str(i))
                             for i in range(0, exp_config.k_clusters)]

        clus_alg = KMeansAlg(
            n_clusters=exp_config.k_clusters,
            random_state=exp_config.random_seed,
            init="k-means++",
            n_init="auto"
        )

        sent_encoder = SentenceTransformerEncoder(
            device=device,
        )

        cluster_label = ClusterLabelMapper(sent_encoder, clus_alg)

    ntp_model = NTPBert(
        checkpoint,
        cluster_label_mapper=cluster_label,

        problem_type="single_label_classification",
        num_labels=len(labels_to_predict),
        label2id={x: i for i, x in enumerate(labels_to_predict)},
        id2label={i: x for i, x in enumerate(labels_to_predict)},

        device=device
    )

    trainer = NTPTrainer(
        ntp_model=ntp_model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        all_labels=all_unique_labels,
        eval_batch_size=eval_batch_size,
        output_name=exp_config.exp_name,
        log_wandb=exp_config.log_wandb,
        train_sampling_fn=sampling_fn,
        monitor_strategy=exp_config.monitor_strategy
    )

    trainer.train(train, val)

    return trainer.output_name
