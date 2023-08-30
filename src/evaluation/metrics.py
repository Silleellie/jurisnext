from __future__ import annotations
from abc import ABC, abstractmethod

from multimethod import multimethod
import numpy as np


class Metric(ABC):

    @abstractmethod
    def __call__(self, predictions: np.ndarray[np.ndarray[str] | str], truths: np.ndarray[str]) -> float:
        raise NotImplementedError


class Hit(Metric):

    def __init__(self, k: int = None):
        self.k = k

    @multimethod
    def __call__(self, predictions: np.ndarray[str], truths: np.ndarray[str]) -> float:

        # k is not used, there's no list to cut
        return np.mean(predictions == truths).item()

    @multimethod
    def __call__(self, predictions: np.ndarray[np.ndarray[str]], truths: np.ndarray[str]) -> float:

        predictions = np.array(predictions)
        truths = np.array(truths)

        predictions = predictions[:, :self.k] if self.k is not None else predictions

        return np.mean(np.isin(truths, predictions)).item()

    def __str__(self):

        string = "Hit"
        if self.k is not None:
            string += f"@{self.k}"

        return string


if __name__ == "__main__":

    metric = Hit()
    metric_cut = Hit(k=1)

    predictions_single = ["a", "b", "c", "d", "e"]
    predictions_multiple = [
        np.array(["f", "b", "e"]),
        np.array(["d", "e", "c"]),
        np.array(["c", "c", "f"]),
        np.array(["f", "f", "a"]),
        np.array(["e", "a", "b"])
    ]
    truth = ["a", "g", "c", "x", "e"]

    predictions_single = np.array(predictions_single)
    predictions_multiple = np.array(predictions_multiple)
    truth = np.array(truth)

    print("OVERLOAD BABY 🔥")
    print(metric(predictions_single, truth))
    print(metric(predictions_multiple, truth))

    print("OVERLOAD BABY CUT 🔥✂️")
    print(metric_cut(predictions_single, truth))
    print(metric_cut(predictions_multiple, truth))
