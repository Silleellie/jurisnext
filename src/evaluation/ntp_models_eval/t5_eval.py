import os

import pandas as pd
from cytoolz import merge_with

from src import MODELS_DIR, METRICS_DIR, ExperimentConfig
from src.data.legal_dataset import LegalDataset
from src.evaluation.metrics import MAP, MRR, Hit
from src.evaluation.ntp_evaluator import NTPEvaluator
from src.evaluation.ntp_models_eval import eval_classification
from src.model.next_title_prediction.ntp_models import NTPT5, DirectNTP, DirectNTPSideInfo, ClusteredNTP, \
    ClusteredNTPSideInfo


def eval_ranking(evaluator, test_set):
    # RANKING
    print("*******************************")
    results = []
    for i, test_split in enumerate(test_set):
        print(f"Evaluating {i+1}-th sampled test set\n")

        result = evaluator.evaluate(
            test_split,
            metrics=[
                Hit(k=10),
                MAP(k=10),
                MRR(k=10),
                Hit(k=5),
                MAP(k=5),
                MRR(k=5),
                Hit(k=3),
                MAP(k=3),
                MRR(k=3),
            ]
        )

        results.append(result)

    # from list of dicts to dict of lists
    results = merge_with(list, *results)

    results_df = pd.DataFrame(results)
    avg_results_df = pd.DataFrame(results_df.mean()).transpose()

    results_df.index = [f"test_split_{i + 1}" for i in range(len(test_set))]
    avg_results_df.index = ["avg_results"]

    print(avg_results_df)
    print("*******************************")

    return avg_results_df, results_df


def t5_eval_main(model_name):

    eval_batch_size = ExperimentConfig.eval_batch_size
    model_pth = os.path.join(MODELS_DIR, model_name)

    ntp_model = NTPT5.load(model_pth)
    ds = LegalDataset.load_dataset()
    test_set = ds.get_hf_datasets()["test"]

    evaluator = NTPEvaluator(ntp_model, eval_batch_size=eval_batch_size)

    test_task_list = [
        DirectNTP(),
        DirectNTPSideInfo(),
        ClusteredNTP(),
        ClusteredNTPSideInfo()
    ]

    # PREDICT ONLY THE NEXT TITLE
    ntp_model.generation_config.num_return_sequences = 1

    classification_task_results = {}

    for test_task in test_task_list:
        print("-------------------------------")
        print(test_task)
        avg_results_df, results_df = eval_classification(evaluator, test_set)
        classification_task_results[test_task] = (results_df, avg_results_df)

    # RANKING eval
    ranking_task_results = {}

    for test_task in test_task_list:
        print("-------------------------------")
        print(test_task)
        avg_results_df, results_df = eval_ranking(evaluator, test_set)
        ranking_task_results[test_task] = (results_df, avg_results_df)

    # SAVE RESULTS in reports/metrics

    os.makedirs(os.path.join(METRICS_DIR, ntp_model.config.name_or_path), exist_ok=True)

    for task_name, (avg_results, all_results) in classification_task_results:

        os.makedirs(os.path.join(METRICS_DIR, model_name, task_name), exist_ok=True)
        all_results.to_csv(os.path.join(METRICS_DIR, model_name, task_name, "classification_all_results.csv"))
        avg_results.to_csv(os.path.join(METRICS_DIR, model_name, task_name, "classification_avg_results.csv"))

    for task_name, (all_results, avg_results) in ranking_task_results:

        os.makedirs(os.path.join(METRICS_DIR, model_name, task_name), exist_ok=True)
        all_results.to_csv(os.path.join(METRICS_DIR, model_name, task_name, "ranking_all_results.csv"))
        avg_results.to_csv(os.path.join(METRICS_DIR, model_name, task_name, "ranking_avg_results.csv"))
