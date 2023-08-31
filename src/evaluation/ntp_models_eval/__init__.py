import pandas as pd
from cytoolz import merge_with

from src.evaluation.metrics import Accuracy, Precision, Recall, F1
from src.evaluation.ntp_evaluator import NTPEvaluator


def eval_classification(evaluator: NTPEvaluator, test_set):
    print("*******************************")
    results = []
    for i, test_split in enumerate(test_set):
        print(f"Evaluating {i+1}-th sampled test set\n")

        result = evaluator.evaluate(
            test_split,
            metrics=[
                Accuracy(),
                Precision(),
                Recall(),
                F1()
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
