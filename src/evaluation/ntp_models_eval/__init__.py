import pandas as pd
import wandb
from cytoolz import merge_with

from src.evaluation.metrics import Accuracy, Precision, Recall, F1
from src.evaluation.ntp_evaluator import NTPEvaluator


def eval_classification(evaluator: NTPEvaluator, test_set,
                        log_wandb: bool = False,
                        prefix_all_metrics: str = "eval_all_metrics",
                        prefix_avg_metrics: str = "eval_avg_metrics"):

    metric_list = [
        Accuracy(),
        Precision(),
        Recall(),
        F1()
    ]

    print("*******************************")
    results = []
    for i, test_split in enumerate(test_set):
        print(f"Evaluating {i+1}-th sampled test set\n")

        result = evaluator.evaluate(
            test_split,
            metrics=metric_list
        )
        results.append(result)

    # from list of dicts to dict of lists
    results = merge_with(list, *results)

    results_df = pd.DataFrame(results)
    avg_results_df = pd.DataFrame(results_df.mean()).transpose()

    results_df.index = [f"test_split_{i + 1}" for i in range(len(test_set))]
    avg_results_df.index = ["avg_results"]

    print("\n\nAverage classification metrics results across all test sets:")
    print(avg_results_df)
    print("*******************************")

    if log_wandb:
        dict_to_log = {
            f"{prefix_all_metrics}/metrics_table": wandb.Table(dataframe=results_df),
        }

        for metric in metric_list:
            dict_to_log[f"{prefix_avg_metrics}/{metric}"] = avg_results_df[str(metric)][0].item()

        wandb.log(dict_to_log)

    return avg_results_df, results_df
