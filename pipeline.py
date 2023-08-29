import argparse

from src.model.next_title_prediction.ntp_models.bert import bert_main
from src.model.next_title_prediction.ntp_models.lm.t5.t5 import t5_main
from src.model.next_title_prediction.ntp_models.multimodal.fusion import multimodal_main
from src.model.next_title_prediction.ntp_models.nli_deberta import nli_deberta_main
from src.utils import seed_everything
from src import ExperimentConfig

available_models_main_func = {
    "t5": t5_main,
    "bert": bert_main,
    "nli_deberta": nli_deberta_main,
    "multimodal": multimodal_main
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
    parser.add_argument('-m', '--model', type=str, default='t5',
                        help='t5 to finetune a t5 checkpoint for Next Title Prediction, '
                             'bert to finetune a bert checkpoint for Next Title Prediction, '
                             'nli_deberta to finetune a deberta checkpoint for Next Title Prediction, '
                             'multimodal to train a multimodal concatenation fusion architecture for Next Title Prediction',
                        metavar='t5')
    parser.add_argument('-ck', '--checkpoint', type=str, default=None,
                        help='Add checkpoint to use for train (e.g. google/flan-t5-small with t5 model)',
                        metavar='None')
    parser.add_argument('--use_clusters', action=argparse.BooleanOptionalAction, default=False,
                        help='Use default clustering algorithm associated with the model during train and eval')

    args = parser.parse_args()

    random_state = args.random_seed
    seed_everything(random_state)

    ExperimentConfig.epochs = args.epochs
    ExperimentConfig.batch_size = args.train_batch_size
    ExperimentConfig.eval_batch_size = args.eval_batch_size
    ExperimentConfig.use_cluster_alg = args.use_clusters
    ExperimentConfig.checkpoint = args.checkpoint

    if args.model in {"t5", "bert", "nli_deberta", "multimodal"}:
        model = args.model
    else:
        raise ValueError("Only 't5', 'bert', 'nli_deberta' or 'multimodal' models are supported!")

    model_main_func = available_models_main_func[model]
    model_main_func()  # each main will use ExperimentConfig parameters
