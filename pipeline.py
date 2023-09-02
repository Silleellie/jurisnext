import argparse
import os
from pathlib import Path

import wandb

from src.data.legal_dataset import data_main
from src.model.next_title_prediction.ntp_models.bert import bert_main, NTPBert
from src.model.next_title_prediction.ntp_models.lm.t5.t5 import t5_main, NTPT5
from src.model.next_title_prediction.ntp_models.multimodal.fusion import multimodal_main
from src.model.next_title_prediction.ntp_models.nli_deberta import nli_deberta_main, NTPNliDeberta

from src.evaluation.ntp_models_eval.t5_eval import t5_eval_main
from src.evaluation.ntp_models_eval.bert_eval import bert_eval_main
from src.evaluation.ntp_models_eval.nli_deberta_eval import nli_deberta_eval_main
from src.evaluation.ntp_models_eval.fusion_eval import multimodal_eval_main

from src.utils import seed_everything, init_wandb
from src import ExperimentConfig, MODELS_DIR

available_models_fns = {
    "t5": (NTPT5.default_checkpoint, t5_main, t5_eval_main),
    "bert": (NTPBert.default_checkpoint, bert_main, bert_eval_main),
    "nli_deberta": (NTPNliDeberta.default_checkpoint, nli_deberta_main, nli_deberta_eval_main),

    # multimodal is not a pretrained model, has no default checkpoint
    "multimodal": ('multimodal_fusion', multimodal_main, multimodal_eval_main)
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main script to run Next Title Prediction experiments')
    parser.add_argument('-epo', '--epochs', type=int, default=100,
                        help='Number of epochs for which the Next Title Prediction model will be trained',
                        metavar='100')
    parser.add_argument('-t_bs', '--train_batch_size', type=int, default=4,
                        help='Batch size that will be used during training',
                        metavar='4')
    parser.add_argument('-e_bs', '--eval_batch_size', type=int, default=2,
                        help='Batch size that will be used during evaluation (validation and testing)',
                        metavar='2')
    parser.add_argument('-seed', '--random_seed', type=int, default=42,
                        help='random seed', metavar='42')
    parser.add_argument('-m', '--model', type=str, default='bert', const='bert', nargs='?',
                        choices=['t5', 'bert', 'nli_deberta', 'multimodal'],
                        help='t5 to finetune a t5 checkpoint on several tasks for Next Title Prediction, '
                             'bert to finetune a bert checkpoint for Next Title Prediction, '
                             'nli_deberta to finetune a deberta checkpoint for Next Title Prediction, '
                             'multimodal to train a multimodal concatenation fusion architecture '
                             'for Next Title Prediction',
                        metavar='bert')
    parser.add_argument('-ck', '--checkpoint', type=str, default=None,
                        help='Add checkpoint to use for train (e.g. google/flan-t5-small with t5 model)',
                        metavar='None')
    parser.add_argument('--use_clusters', action=argparse.BooleanOptionalAction, default=False,
                        help='Use default clustering algorithm associated with the model during train and eval')
    parser.add_argument('--log_wandb', action=argparse.BooleanOptionalAction, default=False,
                        help='Log pipeline information regarding data, train and eval on wandb')
    parser.add_argument('-n_ts', '--n_test_set', type=int, default=10,
                        help='Specify the number of test set to sample for evaluating the model trained',
                        metavar='10')
    parser.add_argument('-d', '--device', type=str, default="cuda:0",
                        help='Specify the device which should be used during the experiment',
                        metavar='cuda:0')
    parser.add_argument('-e', '--exp_name', type=str, default=None,
                        help='Specify a custom name for the trained model which will be saved in the "models" dir',
                        metavar='None')

    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = available_models_fns[args.model][0]

    if args.exp_name is None:
        # replace '/' with '_' to avoid creation of subdir (google/flan-t5-small -> google_flan-t5-small)
        args.exp_name = f"{args.checkpoint.replace('/', '_')}_{args.epochs}"

    if args.log_wandb:

        if 'WANDB_API_KEY' not in os.environ:
            raise ValueError('Cannot log run to wandb if environment variable "WANDB_API_KEY" is not present\n'
                             'Please set the environment variable and add the api key for wandb\n')

        if 'WANDB_ENTITY' not in os.environ:
            raise ValueError('Cannot log run to wandb if environment variable "WANDB_ENTITY" is not present\n'
                             'Please set the environment variable and add the entity for wandb logs\n')

    exp_config = ExperimentConfig(**vars(args))

    # set fixed seed for experiment across all libraries used
    seed_everything(args.random_seed)

    # split dataset and save to disk
    with init_wandb(exp_config.exp_name, 'data', log=exp_config.log_wandb):

        if exp_config.log_wandb:
            wandb.config.update({
                "n_test_set": exp_config.n_test_set,
                "random_seed": exp_config.random_seed,

                # these are hardcoded
                "shuffle": True,
                "split_test_size": 0.2,
                "split_val_size": 0.1
            })

        data_main(exp_config)

    # train model
    model = args.model
    _, model_train_func, model_eval_func = available_models_fns[model]

    with init_wandb(exp_config.exp_name, 'train', log=exp_config.log_wandb):

        if exp_config.log_wandb:
            wandb.config.update({
                "n_epochs": exp_config.epochs,
                "train_batch_size": exp_config.train_batch_size,
                "eval_batch_size": exp_config.eval_batch_size,
                "random_seed": exp_config.random_seed,
                "model": exp_config.model,
                "checkpoint": exp_config.checkpoint,
                "use_clusters": exp_config.use_clusters,
                "device": exp_config.device
            })

        model_name = model_train_func(exp_config)  # each main will use ExperimentConfig instance parameters
        model_path = os.path.join(MODELS_DIR, exp_config.exp_name)

        if exp_config.log_wandb:
            for file in os.listdir(model_path):

                # load various config json of the model fit as artifact
                if Path(file).suffix == ".json":
                    art = wandb.Artifact(name=file, type="hf_config")
                    art.add_file(os.path.join(model_path, file))

                    wandb.log_artifact(art)

    # eval the fit model
    with init_wandb(exp_config.exp_name, 'eval', log=exp_config.log_wandb):

        if exp_config.log_wandb:
            wandb.config.update({
                "n_test_set": exp_config.n_test_set,
                "eval_batch_size": exp_config.eval_batch_size,
                "random_seed": exp_config.random_seed,
            })

        model_eval_func(exp_config)
