import os
from collections import defaultdict

import pandas as pd
import wandb
from cytoolz import merge_with

from src import MODELS_DIR, METRICS_DIR, ExperimentConfig
from src.data.legal_dataset import LegalDataset
from src.evaluation.metrics import MAP, MRR, Hit, Accuracy, Precision, Recall, F1
from src.evaluation.ntp_evaluator import NTPEvaluator
from src.model.next_title_prediction.ntp_models import NTPT5, DirectNTP, DirectNTPSideInfo, ClusteredNTP, \
    ClusteredNTPSideInfo
from src.model.sentence_encoders import SentenceTransformerEncoder


def merge_results(results_list, metric_list, task, prefix_all_metrics, prefix_avg_metrics, log_wandb):
    # from list of dicts to dict of lists
    results = merge_with(list, *results_list)

    results_df = pd.DataFrame(results)
    avg_results_df = pd.DataFrame(results_df.mean()).transpose()

    results_df.index = [f"test_split_{i + 1}" for i in range(0, len(results_list))]
    avg_results_df.index = ["avg_results"]

    print(f"Average metrics results across all test sets for {task}:")
    print(avg_results_df)

    if log_wandb:
        dict_to_log = {
            f"{prefix_all_metrics}/metrics_table": wandb.Table(dataframe=results_df),
        }

        for metric in metric_list:
            dict_to_log[f"{prefix_avg_metrics}/{metric}"] = avg_results_df.iloc[0][str(metric)].item()

        wandb.log(dict_to_log)

    return results_df, avg_results_df


def eval_t5(exp_config, evaluator, test_set, metric_list, prefix):
    task_results = defaultdict(list)

    for i, test_split in enumerate(test_set):
        print(f"Evaluating {i + 1}-th sampled test set\n")
        print("*" * 80)
        print()

        test_task_list = [
            DirectNTP(),
            DirectNTPSideInfo(test_split["input_keywords_sequence"],
                              minimum_occ_number=exp_config.t5_keyword_min_occ)
        ]

        if exp_config.use_clusters:
            test_task_list.append(ClusteredNTP())
            test_task_list.append(ClusteredNTPSideInfo())

        for j, test_task in enumerate(test_task_list):
            print(test_task)
            evaluator.ntp_model.set_test_task(test_task)

            result = evaluator.evaluate(
                test_split,
                metrics=metric_list
            )

            task_results[str(test_task)].append(result)

            # at last we don't want the separator
            if j != len(test_task_list):
                print("-" * 80)

        print()
        print("*" * 80)
        print()

    final_task_results = {}

    print()

    for i, (task, results) in enumerate(task_results.items()):
        final_task_results[task] = merge_results(results, metric_list, task,
                                                 prefix_all_metrics=f"{prefix}/{task}/all_metrics",
                                                 prefix_avg_metrics=f"{prefix}/{task}/avg_metrics",
                                                 log_wandb=exp_config.log_wandb)

        if i != len(task_results):
            print("-" * 80)

    print("#" * 80)

    return final_task_results


def t5_eval_main(exp_config: ExperimentConfig):
    eval_batch_size = exp_config.eval_batch_size
    model_pth = os.path.join(MODELS_DIR, exp_config.exp_name)
    device = exp_config.device

    ds = LegalDataset.load_dataset(exp_config)
    test_set = ds.get_hf_datasets()["test"]
    all_unique_labels = ds.all_unique_labels

    if os.path.isdir(model_pth):
        ntp_model = NTPT5.load(model_pth)
    else:

        sent_encoder = SentenceTransformerEncoder(
            device=device,
        )

        ntp_model = NTPT5(
            exp_config.checkpoint,
            sentence_encoder=sent_encoder,
            training_tasks=None,
            test_task=None,
            all_unique_labels=list(all_unique_labels),
            device=device
        )

        new_words = ['<']

        ntp_model.tokenizer.add_tokens(new_words)
        ntp_model.model.resize_token_embeddings(len(ntp_model.tokenizer))

    evaluator = NTPEvaluator(ntp_model, eval_batch_size=eval_batch_size)

    # RANKING EVALUATION

    print(" RANKING EVALUATION ".center(80, "#"))
    print()

    ranking_metric_list = [
        Hit(k=10),
        MAP(k=10),
        MRR(k=10),
        Hit(k=5),
        MAP(k=5),
        MRR(k=5),
        Hit(k=3),
        MAP(k=3),
        MRR(k=3)
    ]

    ranking_task_results = eval_t5(exp_config, evaluator, test_set, ranking_metric_list, "eval_ranking")

    # PREDICT ONLY THE NEXT TITLE EVALUATION

    print(" CLASSIFICATION EVAL ".center(80, "#"))
    print()

    ntp_model.generation_config.num_return_sequences = 1

    classification_metric_list = [
        Accuracy(),
        Precision(),
        Recall(),
        F1()
    ]

    classification_task_results = eval_t5(exp_config, evaluator, test_set, classification_metric_list,
                                          "eval_classification")

    # SAVE RESULTS in reports/metrics

    os.makedirs(os.path.join(METRICS_DIR, exp_config.exp_name), exist_ok=True)

    for task_name, (all_results, avg_results) in ranking_task_results.items():
        os.makedirs(os.path.join(METRICS_DIR, exp_config.exp_name, task_name), exist_ok=True)
        all_results.to_csv(os.path.join(METRICS_DIR, exp_config.exp_name, task_name, "ranking_all_results.csv"))
        avg_results.to_csv(os.path.join(METRICS_DIR, exp_config.exp_name, task_name, "ranking_avg_results.csv"))

    for task_name, (avg_results, all_results) in classification_task_results.items():
        os.makedirs(os.path.join(METRICS_DIR, exp_config.exp_name, task_name), exist_ok=True)
        all_results.to_csv(os.path.join(METRICS_DIR, exp_config.exp_name, task_name, "classification_all_results.csv"))
        avg_results.to_csv(os.path.join(METRICS_DIR, exp_config.exp_name, task_name, "classification_avg_results.csv"))

    print(f"CSV of the results are saved into {os.path.join(METRICS_DIR, exp_config.exp_name)}!")


if __name__ == "__main__":
    t5_eval_main(ExperimentConfig(model="t5", checkpoint=None, exp_name="google_flan-t5-small_1",
                                  t5_tasks=["directntp", "directntpsideinfo"], pipeline_phases=["eval"],
                                  n_test_set=2))
