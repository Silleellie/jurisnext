import argparse

from src.model.next_title_prediction.ntp_trainer import flan_t5_main, bert_main, deberta_main, multimodal_main
from src.utils import seed_everything

available_models_main_func = {
    "t5": flan_t5_main,
    "bert": bert_main,
    "deberta_nli": deberta_main,
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
    parser.add_argument('-model', '--model', type=str, default='t5',
                        help='t5 to finetune google/flan-t5-small for Next Title Prediction, '
                             'bert to finetune bert-base-uncased for Next Title Prediction, '
                             'deberta_nli to finetune nli-deberta-v3-xsmall for Next Title Prediction, '
                             'multimodal to train a multimodal concatenation fusion architecture for Next Title Prediction',
                        metavar='t5')
    parser.add_argument('--use_clusters', action=argparse.BooleanOptionalAction, default=False,
                        help='Use default clustering algorithm associated with the model during train and eval')

    args = parser.parse_args()

    random_state = args.random_seed
    seed_everything(random_state)

    epochs = args.epochs
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    use_clusters = args.use_clusters

    args_dict = {
        'n_epochs': epochs,
        'batch_size': train_batch_size,
        'eval_batch_size': eval_batch_size,
        'use_cluster_alg': use_clusters
    }

    if args.model in {"t5", "bert", "deberta_nli", "multimodal"}:
        model = args.model
    else:
        raise ValueError("Only 't5', 'bert', 'deberta_nli' or 'multimodal' models are supported!")

    model_main_func = available_models_main_func[model]
    model_main_func(**args_dict)
