import os

from src import MODELS_DIR, METRICS_DIR, ExperimentConfig
from src.data.legal_dataset import LegalDataset
from src.evaluation.ntp_evaluator import NTPEvaluator
from src.evaluation.ntp_models_eval import eval_classification
from src.model.next_title_prediction.ntp_models import NTPBert


def bert_eval_main(exp_config: ExperimentConfig):

    eval_batch_size = exp_config.eval_batch_size
    model_pth = os.path.join(MODELS_DIR, exp_config.exp_name)

    ntp_model = NTPBert.load(model_pth)
    ds = LegalDataset.load_dataset()
    test_set = ds.get_hf_datasets()["test"]

    evaluator = NTPEvaluator(ntp_model, eval_batch_size=eval_batch_size)

    avg_results, all_results = eval_classification(evaluator, test_set)

    os.makedirs(os.path.join(METRICS_DIR, exp_config.exp_name), exist_ok=True)
    all_results.to_csv(os.path.join(METRICS_DIR, exp_config.exp_name, "classification_all_results.csv"))
    avg_results.to_csv(os.path.join(METRICS_DIR, exp_config.exp_name, "classification_avg_results.csv"))
