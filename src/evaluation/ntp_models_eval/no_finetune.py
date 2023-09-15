import os

import datasets
import numpy as np
import pandas as pd
import wandb
from cytoolz import merge_with
from sentence_transformers import util
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import pipeline

from src import ExperimentConfig, METRICS_DIR
from src.data.legal_dataset import LegalDataset
from src.evaluation.metrics import Hit, MRR, MAP, Accuracy, Precision, Recall, F1
from src.model.sentence_encoders import SentenceTransformerEncoder


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def _fill_mask_pipeline(fillmask_model: str,
                        test: datasets.Dataset,
                        all_labels: np.ndarray,
                        top_k: int,
                        device: str,
                        eval_batch_size: int):
    sim_model = SentenceTransformerEncoder(device=device)
    encoded_all_labels = sim_model(*all_labels, show_progress=False, as_tensor=True)

    mask_filler = pipeline("fill-mask",
                           model=fillmask_model,
                           device=device,
                           batch_size=eval_batch_size)

    prep_input = ListDataset([", ".join(title_seq) + ", [MASK]" for title_seq in test["input_title_sequence"]])

    results = [result_dict for result_dict in tqdm(mask_filler(prep_input, top_k=top_k),
                                                   desc="Fill mask preds",
                                                   total=len(prep_input))]

    # we are only interested in the predicted token string
    result_tokens_str = []
    for top_10_list in results:
        top_10_tokens_str = []
        for res_dict in top_10_list:
            top_10_tokens_str.append(res_dict["token_str"])

        result_tokens_str.extend(top_10_tokens_str)

    encoded_preds = sim_model.encode_batch(result_tokens_str)

    sim = util.cos_sim(encoded_preds, encoded_all_labels).cpu()
    mapped_predictions = all_labels[sim.argmax(axis=1)]

    # mapped predictions is 1d. What we want is to have an array of shape (batch_size x num_return sequences)
    mapped_predictions = mapped_predictions.reshape((len(prep_input), top_k))

    return mapped_predictions


def _zeroshot_pipeline(zeroshot_model: str,
                       test: datasets.Dataset,
                       all_labels: np.ndarray,
                       top_k: int,
                       device: str,
                       eval_batch_size: int):

    zeroshot_pipe = pipeline("zero-shot-classification",
                             model=zeroshot_model,
                             device=device,
                             batch_size=eval_batch_size)

    # this prediction are OVER OPTIMISTIC. We consider as candidates labels only those of the test set,
    # but we should consider ALL possible labels of the original
    prep_input = ListDataset([", ".join(title_seq) + ", [MASK]" for title_seq in test["input_title_sequence"]])

    results = [result_dict for result_dict in tqdm(zeroshot_pipe(prep_input, candidate_labels=all_labels.tolist()),
                                                   desc="Zeroshot Preds",
                                                   total=len(prep_input))]

    top_10_lists = [result_dict["labels"][:top_k] for result_dict in results]

    return np.vstack(top_10_lists)


def no_finetune_eval_main(exp_config: ExperimentConfig):
    top_k = 10

    ranking_metric_list = [
        Hit(k=10),
        Hit(k=10),
        MRR(k=10),
        Hit(k=5),
        MAP(k=5),
        MRR(k=5),
        Hit(k=3),
        MAP(k=3),
        MRR(k=3)
    ]

    classification_metric_list = [
        Accuracy(),
        Precision(),
        Recall(),
        F1()
    ]

    ds = LegalDataset.load_dataset(exp_config)
    test_set_list = ds.get_hf_datasets()["test"]

    fill_mask_model = "nlpaueb/legal-bert-base-uncased"
    zeroshot_model = "facebook/bart-large-mnli"

    if exp_config.log_wandb:
        wandb.config.update({"fill_mask_model": fill_mask_model, "zeroshot_model": zeroshot_model})

    print("No finetune models:")
    print(f"fill mask model: {fill_mask_model}")
    print(f"zeroshot_model: {zeroshot_model}")
    print("-" * 80)
    print(f"Evaluating both fill_mask legal and zeroshot model on {len(test_set_list)} test sets...")

    all_ranking_results = {"fill_mask": [], "zeroshot": []}
    all_classification_results = {"fill_mask": [], "zeroshot": []}
    for i, test in enumerate(test_set_list, start=1):

        print(f" {i}/{len(test_set_list)} test set ".center(80, "#"))

        all_labels = np.unique(np.array(test["immediate_next_title"]))

        predictions_fill_mask = _fill_mask_pipeline(fill_mask_model, test, all_labels, top_k,
                                                    exp_config.device, exp_config.eval_batch_size)
        predictions_zeroshot = _zeroshot_pipeline(zeroshot_model, test, all_labels, top_k,
                                                  exp_config.device, exp_config.eval_batch_size)
        truths = np.array(test["immediate_next_title"])

        ranking_eval_mask = {str(metric): metric(predictions_fill_mask, truths)
                             for metric in ranking_metric_list}
        ranking_eval_zeroshot = {str(metric): metric(predictions_zeroshot, truths)
                                 for metric in ranking_metric_list}

        # top-1 prediction for classification metrics, since our truth is one element for ranking list,
        # so if we consider whole ranking list it creates the paradoxical situation where the longer is the
        # ranking list, the lower is the precision
        predictions_fill_mask = predictions_fill_mask[:, 1]
        predictions_zeroshot = predictions_zeroshot[:, 1]

        classification_eval_mask = {str(metric): metric(predictions_fill_mask, truths)
                                    for metric in classification_metric_list}
        classification_eval_zeroshot = {str(metric): metric(predictions_zeroshot, truths)
                                        for metric in classification_metric_list}

        all_ranking_results["fill_mask"].append(ranking_eval_mask)
        all_ranking_results["zeroshot"].append(ranking_eval_zeroshot)

        all_classification_results["fill_mask"].append(classification_eval_mask)
        all_classification_results["zeroshot"].append(classification_eval_zeroshot)

    # RANKING METRICS
    # from list of dicts to dict of lists
    all_ranking_results["fill_mask"] = merge_with(list, *all_ranking_results["fill_mask"])
    all_ranking_results["zeroshot"] = merge_with(list, *all_ranking_results["zeroshot"])

    mask_ranking_all_results_df = pd.DataFrame(all_ranking_results["fill_mask"])
    mask_ranking_avg_results_df = pd.DataFrame(mask_ranking_all_results_df.mean()).transpose()

    mask_ranking_all_results_df.index = [f"test_split_{i + 1}" for i in range(len(test_set_list))]
    mask_ranking_avg_results_df.index = ["avg_results"]

    zeroshot_ranking_all_results_df = pd.DataFrame(all_ranking_results["zeroshot"])
    zeroshot_ranking_avg_results_df = pd.DataFrame(zeroshot_ranking_all_results_df.mean()).transpose()

    zeroshot_ranking_all_results_df.index = [f"test_split_{i + 1}" for i in range(len(test_set_list))]
    zeroshot_ranking_avg_results_df.index = ["avg_results"]

    print("\n\nFILL_MASK MODEL: Average ranking metrics results across all test sets:")
    print(mask_ranking_avg_results_df)
    print("-" * 80)
    print("ZEROSHOT MODEL: Average ranking metrics results across all test sets:")
    print(zeroshot_ranking_avg_results_df)
    print("*" * 80)

    # CLASSIFICATION METRICS
    # from list of dicts to dict of lists
    all_classification_results["fill_mask"] = merge_with(list, *all_classification_results["fill_mask"])
    all_classification_results["zeroshot"] = merge_with(list, *all_classification_results["zeroshot"])

    mask_classification_all_results_df = pd.DataFrame(all_classification_results["fill_mask"])
    mask_classification_avg_results_df = pd.DataFrame(mask_classification_all_results_df.mean()).transpose()

    mask_classification_all_results_df.index = [f"test_split_{i + 1}" for i in range(len(test_set_list))]
    mask_classification_avg_results_df.index = ["avg_results"]

    zeroshot_classification_all_results_df = pd.DataFrame(all_classification_results["zeroshot"])
    zeroshot_classification_avg_results_df = pd.DataFrame(zeroshot_classification_all_results_df.mean()).transpose()

    zeroshot_classification_all_results_df.index = [f"test_split_{i + 1}" for i in range(len(test_set_list))]
    zeroshot_classification_avg_results_df.index = ["avg_results"]

    print("\n\nFILL_MASK MODEL: Average classification metrics results across all test sets:")
    print(mask_classification_avg_results_df)
    print("-" * 80)
    print("ZEROSHOT MODEL: Average classification metrics results across all test sets:")
    print(zeroshot_classification_avg_results_df)
    print("*" * 80)

    # SAVE RESULTS in reports/metrics

    os.makedirs(os.path.join(METRICS_DIR, exp_config.exp_name, "fill_mask"), exist_ok=True)
    os.makedirs(os.path.join(METRICS_DIR, exp_config.exp_name, "zeroshot"), exist_ok=True)

    mask_ranking_all_results_df.to_csv(
        os.path.join(METRICS_DIR, exp_config.exp_name, "fill_mask", "ranking_all_results.csv")
    )
    mask_ranking_avg_results_df.to_csv(
        os.path.join(METRICS_DIR, exp_config.exp_name, "fill_mask", "ranking_avg_results.csv")
    )
    zeroshot_ranking_all_results_df.to_csv(
        os.path.join(METRICS_DIR, exp_config.exp_name, "zeroshot", "ranking_all_results.csv")
    )
    zeroshot_ranking_avg_results_df.to_csv(
        os.path.join(METRICS_DIR, exp_config.exp_name, "zeroshot", "ranking_avg_results.csv")
    )

    mask_classification_all_results_df.to_csv(
        os.path.join(METRICS_DIR, exp_config.exp_name, "fill_mask", "classification_all_results.csv")
    )
    mask_classification_avg_results_df.to_csv(
        os.path.join(METRICS_DIR, exp_config.exp_name, "fill_mask", "classification_avg_results.csv")
    )
    zeroshot_classification_all_results_df.to_csv(
        os.path.join(METRICS_DIR, exp_config.exp_name, "zeroshot", "classification_all_results.csv")
    )
    zeroshot_classification_avg_results_df.to_csv(
        os.path.join(METRICS_DIR, exp_config.exp_name, "zeroshot", "classification_avg_results.csv")
    )

    if exp_config.log_wandb:

        dict_to_log = {
            "fill_mask/eval_ranking/all_metrics": wandb.Table(dataframe=mask_ranking_all_results_df),
            "zeroshot/eval_ranking/all_metrics": wandb.Table(dataframe=zeroshot_ranking_all_results_df),
            "fill_mask/eval_classification/all_metrics": wandb.Table(dataframe=mask_classification_all_results_df),
            "zeroshot/eval_classification/all_metrics": wandb.Table(dataframe=zeroshot_classification_all_results_df)
        }

        for metric in ranking_metric_list:
            dict_to_log[f"fill_mask/eval_ranking/avg_metrics/{metric}"] = mask_ranking_avg_results_df.iloc[0][
                str(metric)].item()
            dict_to_log[f"zeroshot/eval_ranking/avg_metrics/{metric}"] = zeroshot_ranking_avg_results_df.iloc[0][
                str(metric)].item()

        for metric in classification_metric_list:
            dict_to_log[f"fill_mask/eval_classification/avg_metrics/{metric}"] = \
            mask_classification_avg_results_df.iloc[0][str(metric)].item()
            dict_to_log[f"zeroshot/eval_classification/avg_metrics/{metric}"] = \
            zeroshot_classification_avg_results_df.iloc[0][str(metric)].item()

        wandb.log(dict_to_log)

    print(f"CSV of the results are saved into {os.path.join(METRICS_DIR, exp_config.exp_name)}!")


if __name__ == "__main__":

    no_finetune_eval_main(ExperimentConfig(model="no_finetune"))
