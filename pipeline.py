import argparse
import dataclasses
import os

import wandb

from src.data.legal_dataset import data_main
from src.evaluation.ntp_models_eval.cnn_eval import cnn_eval_main
from src.evaluation.ntp_models_eval.lstm_eval import lstm_eval_main
from src.evaluation.ntp_models_eval.no_finetune import no_finetune_eval_main
from src.model.next_title_prediction.ntp_models.bert import bert_main, NTPBert
from src.model.next_title_prediction.ntp_models.custom_encoders.cnn_encoder_model import cnn_model_main
from src.model.next_title_prediction.ntp_models.custom_encoders.lstm_encoder_model import lstm_model_main
from src.model.next_title_prediction.ntp_models.lm.t5.t5 import t5_main, NTPT5
from src.model.next_title_prediction.ntp_models.custom_encoders.fusion import fusion_main
from src.model.next_title_prediction.ntp_models.nli_deberta import nli_deberta_main, NTPNliDeberta

from src.evaluation.ntp_models_eval.t5_eval import t5_eval_main
from src.evaluation.ntp_models_eval.bert_eval import bert_eval_main
from src.evaluation.ntp_models_eval.nli_deberta_eval import nli_deberta_eval_main
from src.evaluation.ntp_models_eval.fusion_eval import fusion_eval_main

from src.utils import seed_everything, init_wandb
from src import ExperimentConfig, MODELS_DIR

available_models_fns = {
    "t5": (NTPT5.default_checkpoint, t5_main, t5_eval_main),
    "bert": (NTPBert.default_checkpoint, bert_main, bert_eval_main),
    "nli_deberta": (NTPNliDeberta.default_checkpoint, nli_deberta_main, nli_deberta_eval_main),

    # fusion is not a pretrained model, has no default checkpoint
    "fusion": ('fusion', fusion_main, fusion_eval_main),
    "lstm": ('lstm', lstm_model_main, lstm_eval_main),
    "cnn": ('cnn', cnn_model_main, cnn_eval_main),

    "no_finetune": ('no_finetune', None, no_finetune_eval_main)
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
    parser.add_argument('-monitor', '--monitor_strategy', type=str, default='loss', const='loss', nargs='?',
                        choices=['loss', 'metric'],
                        help='Choose the strategy used to save the best model. If "loss", the validation loss will be '
                             'used to save the best model, if "metric", the reference metric (Accuracy weighted or '
                             'Hit) will be used to save the best model',
                        metavar='loss')
    parser.add_argument('-m', '--model', type=str, default='bert', const='bert', nargs='?', required=True,
                        choices=['t5', 'bert', 'nli_deberta', 'fusion', 'lstm', 'cnn', 'no_finetune'],
                        help='t5 to finetune a t5 checkpoint on several tasks for Next Title Prediction, '
                             'bert to finetune a bert checkpoint for Next Title Prediction, '
                             'nli_deberta to finetune a deberta checkpoint for Next Title Prediction, '
                             'multimodal to train a multimodal concatenation fusion architecture '
                             'for Next Title Prediction',
                        metavar='bert')
    parser.add_argument('-ck', '--checkpoint', type=str, default=None,
                        help='Add checkpoint to use for train (e.g. google/flan-t5-small with t5 model)',
                        metavar='None')
    parser.add_argument('-k_c', '--k_clusters', type=int, default=None,
                        help='If specified, it sets the number of clustered labels that will be considered '
                             'as next possible titles instead of the original labels',
                        metavar='None')
    parser.add_argument('--log_wandb', action=argparse.BooleanOptionalAction, default=False,
                        help='Log pipeline information regarding data, train and eval on wandb')
    parser.add_argument('-n_ts', '--n_test_set', type=int, default=10,
                        help='Specify the number of test set to sample for evaluating the model trained',
                        metavar='10')
    parser.add_argument('-p_s', '--prediction_supporter', type=str, default=None,
                        help='Specify the name of the folder in the models directory containing the model to use as '
                             'prediction supporter',
                        metavar='None')
    parser.add_argument('-t5_t', '--t5_tasks', nargs="+", default=None,
                        choices=["directNTP", "directNTPSideInfo", "boolNTP"],
                        help='Specify which train task to use to fine tune NTPT55. If not specified, all possible tasks'
                             ' will be used. The first task specified will be used as validation task. '
                             'In the eval phase, all possible tasks are evaluated (apart from BoolNTP which is a '
                             'support task)',
                        metavar='None')
    parser.add_argument('-t5_kw_min', '--t5_keyword_min_occ', type=int, default=None,
                        help='Specify what is the min occurrences that a keyword should have in order to be picked as '
                             'side info. If not specified, keywords will be sampled randomly',
                        metavar='None')
    parser.add_argument('-ngram', '--ngram_label', type=int, default=None,
                        help='Specify the max number of ngram that a label can have. If None, all ngrams are used',
                        metavar='None')
    parser.add_argument('-seq_sampling', '--seq_sampling_strategy', type=str, default="random",
                        choices=['random', 'augment'],
                        help='Specify how sampling is performed on the dataset. "random" will consider randomly '
                             'picked sequences, "augment" will consider all possible incremental sequences',
                        metavar='random')
    parser.add_argument('-seq_sampling_start', '--seq_sampling_start_strategy', type=str, default="beginning",
                        choices=['beginning', 'random'],
                        help='Specify how the sampled sequence should start. This parameter is ignored if '
                             '--seq_sampling_strategy is set to "augment". "beginning" will sample the sequence '
                             'starting always from the first element, "random" will sample also the starting point '
                             'of the sampled sequence',
                        metavar='beginning')
    parser.add_argument('-test_seq_sampling', '--test_seq_sampling_strategy', type=str, default=None,
                        choices=['random', 'augment'],
                        help='Specify how sampling is performed on the test set. "random" will consider randomly '
                             'picked sequences, "augment" will consider all possible incremental sequences. '
                             'If not set, the sampling strategy of train/val will be used. If this parameter is '
                             'set to "augment" and --n_test_set > 1, a warning is printed and n_test_set is forced '
                             'to 1',
                        metavar='None')
    parser.add_argument('-clean_kdws', '--clean_stopwords_kwds', action=argparse.BooleanOptionalAction, default=False,
                        help='Specify whether to remove stopwords from the keywords column of the dataset or not')
    parser.add_argument('-d', '--device', type=str, default="cuda:0",
                        help='Specify the device which should be used during the experiment',
                        metavar='cuda:0')
    parser.add_argument('-e', '--exp_name', type=str, default=None,
                        help='Specify a custom name for the trained model which will be saved in the "models" dir',
                        metavar='None')
    parser.add_argument('-phase', '--pipeline_phases', nargs="+", default=None,
                        choices=["data", "train", "eval"],
                        help='If specified, only the selected part(s) of the pipeline are carried out. By default, all '
                             'phases are performed',
                        metavar='None')

    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = available_models_fns[args.model][0]

    if args.exp_name is None:
        # replace '/' with '_' to avoid creation of subdir (google/flan-t5-small -> google_flan-t5-small)
        args.exp_name = f"{args.checkpoint.replace('/', '_')}_{args.epochs}"

    if args.t5_tasks is None:
        args.t5_tasks = ["directNTP", "directNTPSideInfo", "boolNTP"]

    # lowercase conversion to standardize
    args.t5_tasks = [task_name.lower() for task_name in args.t5_tasks]

    if args.pipeline_phases is None:
        args.pipeline_phases = ["data", "train", "eval"]

    if args.log_wandb:

        if 'WANDB_API_KEY' not in os.environ:
            raise ValueError('Cannot log run to wandb if environment variable "WANDB_API_KEY" is not present\n'
                             'Please set the environment variable and add the api key for wandb\n')

        if 'WANDB_ENTITY' not in os.environ:
            raise ValueError('Cannot log run to wandb if environment variable "WANDB_ENTITY" is not present\n'
                             'Please set the environment variable and add the entity for wandb logs\n')

    exp_config = ExperimentConfig(**vars(args))

    print("Experiment configuration:")
    print(dataclasses.asdict(exp_config))

    # set fixed seed for experiment across all libraries used
    seed_everything(args.random_seed)

    if args.model == "no_finetune":
        print("'No fine tune' experiment chosen, only evaluation phase will be performed!")

    # DATA PIPELINE
    if 'data' in exp_config.pipeline_phases:

        # split dataset and save to disk
        with init_wandb(exp_config.exp_name, 'data', log=exp_config.log_wandb):

            if exp_config.log_wandb:
                wandb.config.update({
                    "n_test_set": exp_config.n_test_set,
                    "random_seed": exp_config.random_seed,
                    "ngram_label": exp_config.ngram_label,
                    "seq_sampling_strategy": exp_config.seq_sampling_strategy,
                    "seq_sampling_start_strategy": exp_config.seq_sampling_start_strategy,
                    "test_seq_sampling_strategy": exp_config.test_seq_sampling_strategy,
                    "clean_stopwords_kwds": exp_config.clean_stopwords_kwds,

                    # these are hardcoded
                    "shuffle": True,
                    "split_test_size": 0.2,
                    "split_val_size": 0.1
                })

            data_main(exp_config)

    # TRAIN PIPELINE: if "no_finetune" experiment, we skip this
    if 'train' in exp_config.pipeline_phases and args.model != "no_finetune":
        model = args.model
        _, model_train_func, _ = available_models_fns[model]

        with init_wandb(exp_config.exp_name, 'train', log=exp_config.log_wandb):

            if exp_config.log_wandb:

                wandb_dict = {
                    "n_epochs": exp_config.epochs,
                    "train_batch_size": exp_config.train_batch_size,
                    "eval_batch_size": exp_config.eval_batch_size,
                    "random_seed": exp_config.random_seed,
                    "model": exp_config.model,
                    "checkpoint": exp_config.checkpoint,
                    "device": exp_config.device
                }

                if args.model == "t5":
                    wandb_dict["t5_tasks"] = exp_config.t5_tasks
                    wandb_dict["t5_keyword_min_occ"] = exp_config.t5_keyword_min_occ

                wandb.config.update(wandb_dict)

            model_name = model_train_func(exp_config)  # each main will use ExperimentConfig instance parameters
            model_path = os.path.join(MODELS_DIR, exp_config.exp_name)

    # before evaluating, set fix seed so that even across
    # separate runs of train and eval results are reproducible
    seed_everything(args.random_seed)

    # EVAL PIPELINE
    if 'eval' in exp_config.pipeline_phases:

        model = args.model
        _, _, model_eval_func = available_models_fns[model]

        with init_wandb(exp_config.exp_name, 'eval', log=exp_config.log_wandb):

            if exp_config.log_wandb:
                wandb.config.update({
                    "n_test_set": exp_config.n_test_set,
                    "ngram_label": exp_config.ngram_label,
                    "eval_batch_size": exp_config.eval_batch_size,
                    "random_seed": exp_config.random_seed,
                })

            model_eval_func(exp_config)
