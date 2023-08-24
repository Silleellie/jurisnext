import argparse

from src.model.sequence_classification.seq_trainer import flan_t5_main, multimodal_main
from src.utils import seed_everything

available_models_main_func = {
    "t5": flan_t5_main,
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
                             'multimodal to train a multimodal concatenation fusion architecture for Next Title Prediction',
                        metavar='t5')

    args = parser.parse_args()

    random_state = args.random_seed
    seed_everything(random_state)

    epochs = args.epochs
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size

    args_dict = {
        'n_epochs': epochs,
        'batch_size': train_batch_size,
        'eval_batch_size': eval_batch_size
    }

    # to do: add bert and deberta
    if args.model in {"t5", "multimodal"}:
        model = args.model
    else:
        raise ValueError("Only 't5' or 'multimodal' models are supported!")

    model_main_func = available_models_main_func[model]
    model_main_func(**args_dict)
