import os.path
from math import ceil
from typing import Callable, List

import datasets
from tqdm import tqdm
import numpy as np

from src import MODELS_DIR
from src.data.legal_dataset import LegalDataset
from src.evaluation.metrics import Metric, Hit
from src.model.next_title_prediction.ntp_models import NTPT5
from src.model.next_title_prediction.ntp_models_abtract import NTPModel


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
        preprocessed_test = test_dataset.map(self.ntp_model.tokenize,
                                             remove_columns=test_dataset.column_names)
        preprocessed_test.set_format("torch")

        total_n_batch = ceil(preprocessed_test.num_rows / self.eval_batch_size)

        pbar_test = tqdm(preprocessed_test.iter(batch_size=self.eval_batch_size),
                         total=total_n_batch)

        total_preds = []
        total_truths = []

        for i, batch in enumerate(pbar_test, start=1):

            prepared_input = self.ntp_model.prepare_input(batch)
            predictions, truths, _ = self.ntp_model.valid_step(prepared_input)

            total_preds.extend(predictions)
            total_truths.extend(truths)

            # we update the loss every 1% progress considering the total n° of batches
            if (i % ceil(total_n_batch / 100)) == 0:

                n_total_pred_so_far = len(total_preds)
                matches = (np.array(total_preds) == np.array(total_truths)).sum()

                pbar_test.set_description(f"Acc -> {(matches / n_total_pred_so_far):.3f}")

        total_preds = np.array(total_preds)
        total_truths = np.array(total_truths)

        res_eval_dict = {str(metric): metric(total_preds, total_truths)
                         for metric in metrics}

        return res_eval_dict


if __name__ == "__main__":

    model_pth = os.path.join(MODELS_DIR, "google/flan-t5-small_20")

    ntp_model = NTPT5.load(model_pth)
    ds = LegalDataset.load_dataset()
    test_set = ds.get_hf_datasets()["test"]

    evaluator = NTPEvaluator(ntp_model, eval_batch_size=2)
    result = evaluator.evaluate(test_set, metrics=[Hit(), Hit(k=1)])

    print(result)
