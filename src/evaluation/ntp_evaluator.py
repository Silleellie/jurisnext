from math import ceil
from typing import List

import datasets
from tqdm import tqdm
import numpy as np

from src.evaluation.metrics import Metric
from src.model.next_title_prediction.ntp_models_abtract import NTPModel
from src.utils import add_cluster_column


class NTPEvaluator:

    def __init__(self,
                 ntp_model: NTPModel,
                 eval_batch_size: int = 4):

        self.ntp_model = ntp_model
        self.eval_batch_size = eval_batch_size

    def evaluate(self,
                 test_dataset: datasets.Dataset,
                 metrics: List[Metric]):

        self.ntp_model.eval()

        if self.ntp_model.prediction_supporter is not None:
            test_dataset = add_cluster_column(self.ntp_model, test_dataset, "test set", self.eval_batch_size)

        preprocessed_test = test_dataset.map(self.ntp_model.tokenize,
                                             remove_columns=test_dataset.column_names,
                                             load_from_cache_file=False,
                                             desc="Tokenizing test set")
        preprocessed_test.set_format("torch")

        total_n_batch = ceil(preprocessed_test.num_rows / self.eval_batch_size)

        pbar_test = tqdm(preprocessed_test.iter(batch_size=self.eval_batch_size),
                         total=total_n_batch)

        total_preds = []
        total_truths = []

        pbar_test.set_description("Computing predictions for eval...")

        for i, batch in enumerate(pbar_test, start=1):

            prepared_input = self.ntp_model.prepare_input(batch)
            predictions, truths, _ = self.ntp_model.valid_step(prepared_input)

            total_preds.extend(predictions)
            total_truths.extend(truths)

        total_preds = np.array(total_preds).squeeze()
        total_truths = np.array(total_truths)

        res_eval_dict = {str(metric): metric(total_preds, total_truths)
                         for metric in metrics}

        return res_eval_dict
