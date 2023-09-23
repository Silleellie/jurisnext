import random
from typing import List, Dict, Any

import numpy as np
import torch
from sentence_transformers import util
from torch.nn.utils.rnn import pad_sequence
from transformers import T5ForConditionalGeneration, Adafactor, T5Config, GenerationConfig

from src import ExperimentConfig
from src.data.legal_dataset import LegalDataset
from src.model.next_title_prediction.ntp_models.lm.t5.templates import DirectNTP, BoolNTP, Task, DirectNTPSideInfo
from src.model.next_title_prediction.ntp_trainer import NTPTrainer
from src.model.sentence_encoders import SentenceEncoder, SentenceTransformerEncoder
from src.model.next_title_prediction.ntp_models_abtract import NTPConfig, NTPModelHF


class NTPT5Config(NTPConfig, T5Config):

    def __init__(self,
                 training_tasks: List[Task] = None,
                 test_task: Task = DirectNTP(),
                 all_unique_labels: List[str] = None,
                 device: str = "cpu",
                 **kwargs):

        NTPConfig.__init__(self, device)
        T5Config.__init__(self, **kwargs)

        self.training_tasks = training_tasks if training_tasks is not None else []
        self.test_task = test_task
        self.all_unique_labels = np.array(all_unique_labels) if all_unique_labels is not None else np.array([])
        self.device = device

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):

        training_tasks = kwargs.pop("training_tasks", None)
        test_task = kwargs.pop("test_task", None)
        all_unique_labels = kwargs.pop("all_unique_labels", None)

        # training_tasks is a mandatory attribute, it's either in config_dict or kwargs
        if training_tasks is not None:
            config_dict["training_tasks"] = training_tasks
        else:
            eval_training_tasks = []
            for train_task_class, train_task_parameters in config_dict["training_tasks"]:
                eval_task = Task.from_eval(train_task_class, train_task_parameters)
                eval_training_tasks.append(eval_task)

            config_dict["training_tasks"] = eval_training_tasks

        # test_task is not mandatory, that's why "elif"
        if test_task is not None:
            config_dict["test_task"] = test_task
        elif "test_task" in config_dict:
            test_task_class, test_task_parameters = config_dict["test_task"]
            config_dict["test_task"] = Task.from_eval(test_task_class, test_task_parameters)

        if all_unique_labels is not None:
            config_dict["all_unique_labels"] = all_unique_labels

        return super().from_dict(config_dict, **kwargs)

    # to make __repr__ work we need to convert the tensor to a json serializable format
    def to_dict(self) -> Dict[str, Any]:
        super_dict = super().to_dict()

        super_dict["training_tasks"] = [task.to_json() for task in super_dict["training_tasks"]]
        super_dict["test_task"] = super_dict["test_task"].to_json()
        super_dict["all_unique_labels"] = list(super_dict["all_unique_labels"])

        return super_dict


class NTPT5(NTPModelHF):
    model_class = T5ForConditionalGeneration
    config_class = NTPT5Config
    default_checkpoint = 'google/flan-t5-small'

    def __init__(self,
                 pretrained_model_or_pth: str = default_checkpoint,
                 sentence_encoder: SentenceEncoder = SentenceTransformerEncoder(),
                 **config_and_gen_kwargs):

        # to avoid duplicate parameter error
        config_and_gen_kwargs.pop("return_unused_kwargs", None)

        config_and_gen_kwargs["num_return_sequences"] = config_and_gen_kwargs.pop("num_return_sequences", 10)
        config_and_gen_kwargs["max_new_tokens"] = config_and_gen_kwargs.pop("max_new_tokens", 50)
        config_and_gen_kwargs["num_beams"] = config_and_gen_kwargs.pop("num_beams", 30)
        config_and_gen_kwargs["no_repeat_ngram_size"] = config_and_gen_kwargs.pop("no_repeat_ngram_size", 0)
        config_and_gen_kwargs["early_stopping"] = config_and_gen_kwargs.pop("early_stopping", True)

        self.generation_config, config_kwargs = GenerationConfig.from_pretrained(
            pretrained_model_or_pth, return_unused_kwargs=True, **config_and_gen_kwargs
        )

        super().__init__(
            pretrained_model_or_pth=pretrained_model_or_pth,
            **config_kwargs
        )

        self.sim_model = sentence_encoder
        self.encoded_all_labels = self.sim_model(*self.config.all_unique_labels,
                                                 desc="Encoding ALL labels for FlanT5...",
                                                 as_tensor=True)

    def get_suggested_optimizer(self):
        return Adafactor(
            list(self.model.parameters()),
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

    def set_test_task(self, test_task: Task):
        self.config.test_task = test_task

    def tokenize(self, sample):

        title_sequence = sample["input_title_sequence"]
        next_title = sample["immediate_next_title"]
        rel_keywords_sequence = sample["input_keywords_sequence"]

        task = random.choice(self.config.training_tasks) if self.training else self.config.test_task

        input_text, target_text = task(title_sequence, next_title,
                                       rel_keywords_seq=rel_keywords_sequence)

        encoded_sequence = self.tokenizer(text=input_text, text_target=target_text, truncation=True)
        encoded_sequence["immediate_next_title"] = next_title

        return encoded_sequence

    def prepare_input(self, batch):
        input_dict = {}

        input_ids = pad_sequence(batch["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(batch["attention_mask"], batch_first=True,
                                      padding_value=self.tokenizer.pad_token_id)

        input_dict["input_ids"] = input_ids.to(self.config.device)
        input_dict["attention_mask"] = attention_mask.to(self.config.device)

        if "labels" in batch:
            lm_labels = pad_sequence(batch["labels"], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100

            input_dict["labels"] = lm_labels.to(self.config.device)

        if not self.training:
            input_dict["immediate_next_title"] = batch["immediate_next_title"]

        return input_dict

    def train_step(self, batch):

        output = self(**batch)

        return output.loss

    @torch.no_grad()
    def valid_step(self, batch):

        target_text = batch.pop("immediate_next_title")
        output = self(**batch)

        beam_outputs = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            generation_config=self.generation_config
        )

        generated_sents = self.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
        encoded_preds = self.sim_model.encode_batch(generated_sents)

        sim = util.cos_sim(encoded_preds, self.encoded_all_labels).cpu()
        mapped_predictions = self.config.all_unique_labels[sim.argmax(axis=1)]

        # mapped predictions is 1d. What we want is to have an array of shape (batch_size x num_return sequences)
        mapped_predictions = mapped_predictions.reshape((len(target_text), self.generation_config.num_return_sequences))

        val_loss = output.loss

        return mapped_predictions, target_text, val_loss


def t5_main(exp_config: ExperimentConfig):
    n_epochs = exp_config.epochs
    batch_size = exp_config.train_batch_size
    eval_batch_size = exp_config.eval_batch_size
    device = exp_config.device

    checkpoint = "google/flan-t5-small"
    if exp_config.checkpoint is not None:
        checkpoint = exp_config.checkpoint

    ds = LegalDataset.load_dataset(exp_config)
    dataset = ds.get_hf_datasets()
    all_unique_labels = ds.all_unique_labels

    train = dataset["train"]
    val = dataset["validation"]
    sampling_fn = ds.perform_sampling

    all_rel_keywords = train["rel_keywords_sequence"] + val["input_keywords_sequence"]

    train_task_list = []

    if exp_config.t5_tasks == ['boolntp']:
        raise ValueError("Please insert one other task other than only boolNTP! boolNTP it's only a support task "
                         "and does not compute useful predictions for the ntp task")

    test_task = None
    if 'directntp' in exp_config.t5_tasks:
        task = DirectNTP()
        train_task_list.append(task)

        if exp_config.t5_tasks[0] == "directntp":
            test_task = task
    if 'directntpsideinfo' in exp_config.t5_tasks:
        task = DirectNTPSideInfo(all_rel_keywords, minimum_occ_number=exp_config.t5_keyword_min_occ)
        train_task_list.append(task)

        if exp_config.t5_tasks[0] == "directntpsideinfo":
            test_task = task
    if 'boolntp' in exp_config.t5_tasks:
        train_task_list.append(BoolNTP(all_unique_labels))

    if test_task is None:
        print(f"boolNTP set as first task of the --t5_tasks parameter, it is ignored. {train_task_list[0]} will be "
              f"used instead as validation task!")
        test_task = train_task_list[0]

    print(test_task)
    sent_encoder = SentenceTransformerEncoder(
        device=device,
    )

    model_ntp = NTPT5(
        checkpoint,
        sentence_encoder=sent_encoder,

        training_tasks=train_task_list,
        test_task=test_task,
        all_unique_labels=list(all_unique_labels),
        device=device
    )

    new_words = ['<']

    model_ntp.tokenizer.add_tokens(new_words)
    model_ntp.model.resize_token_embeddings(len(model_ntp.tokenizer))

    trainer = NTPTrainer(
        ntp_model=model_ntp,
        n_epochs=n_epochs,
        batch_size=batch_size,
        all_labels=all_unique_labels,
        eval_batch_size=eval_batch_size,
        output_name=exp_config.exp_name,
        log_wandb=exp_config.log_wandb,
        random_seed=exp_config.random_seed,
        train_sampling_fn=sampling_fn,
        monitor_strategy=exp_config.monitor_strategy
    )

    trainer.train(train, val)

    return trainer.output_name


if __name__ == "__main__":
    vars = {'model': 't5', 'checkpoint': 'google/flan-t5-small', 'exp_name': 'flan_t5_ablation_direct',
            'pipeline_phases': ['data', 'train', 'eval'], 'epochs': 200, 'train_batch_size': 2, 'eval_batch_size': 2,
            'random_seed': 42, 'monitor_strategy': 'loss', 'use_clusters': False, 'log_wandb': False, 'n_test_set': 10,
            'ngram_label': None, 'seq_sampling_strategy': 'random', 'clean_stopwords_kwds': False,
            't5_keyword_min_occ': None,
            't5_tasks': ['directntp'], 'device': 'cuda:0'}

    exp_config = ExperimentConfig(**vars)

    t5_main(exp_config)
