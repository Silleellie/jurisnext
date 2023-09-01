import argparse

from src.data.legal_dataset import data_main
from src.model.next_title_prediction.ntp_models.bert import bert_main
from src.model.next_title_prediction.ntp_models.lm.t5.t5 import t5_main
from src.model.next_title_prediction.ntp_models.multimodal.fusion import multimodal_main
from src.model.next_title_prediction.ntp_models.nli_deberta import nli_deberta_main

from src.evaluation.ntp_models_eval.t5_eval import t5_eval_main
from src.evaluation.ntp_models_eval.bert_eval import bert_eval_main
from src.evaluation.ntp_models_eval.nli_deberta_eval import nli_deberta_eval_main
from src.evaluation.ntp_models_eval.fusion_eval import multimodal_eval_main

from src.utils import seed_everything
from src import ExperimentConfig

available_models_main_func = {
    "t5": (t5_main, t5_eval_main),
    "bert": (bert_main, bert_eval_main),
    "nli_deberta": (nli_deberta_main, nli_deberta_eval_main),
    "multimodal": (multimodal_main, multimodal_eval_main)
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main script to run Next Title Prediction experiments')
    parser.add_argument('-epo', '--epochs', type=int, default=100,
                        help='Number of epochs for which the Next Title Prediction model will be trained', metavar='100')
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
    parser.add_argument('-n_ts', '--n_test_set', type=int, default=10,
                        help='Specify the number of test set to sample for evaluating the model trained',
                        metavar='10')
    parser.add_argument('-d', '--device', type=str, default="cuda:0",
                        help='Specify the device which should be used during the experiment',
                        metavar='cuda:0')
    parser.add_argument('-o', '--output_name', type=str, default=None,
                        help='Specify a custom name for the trained model which will be saved in the "models" dir',
                        metavar='None')

    args = parser.parse_args()

    ExperimentConfig.epochs = args.epochs
    ExperimentConfig.batch_size = args.train_batch_size
    ExperimentConfig.eval_batch_size = args.eval_batch_size
    ExperimentConfig.random_state = args.random_seed
    ExperimentConfig.checkpoint = args.checkpoint
    ExperimentConfig.use_cluster_alg = args.use_clusters
    ExperimentConfig.n_test_set = args.n_test_set
    ExperimentConfig.output_name = args.output_name

    # set fixed seed for experiment across all libraries used
    seed_everything(args.random_seed)

    # split dataset and save to disk
    data_main()

    # train model
    model = args.model
    model_train_func, model_eval_func = available_models_main_func[model]
    model_name = model_train_func()  # each main will use ExperimentConfig parameters

    # eval the fit model
    model_eval_func(model_name)
