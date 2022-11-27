# -*- encoding: utf-8 -*-

# Standard Imports
import random
import pickle
import inspect
from pathlib import Path

# Third Party Imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local import
from train_embedding import Embedding, Key

# Typing Imports
from typing import Optional, Sequence, Callable, Any

import warnings

warnings.filterwarnings("ignore")


def metric(func: Callable) -> Callable:
    func.ismetric = True
    return func


class Model:
    def __init__(self, input_path: Optional[str | Path] = None, threshold: float = 0.3):
        self.embedding = Embedding.load(input_path)
        self.threshold = threshold
        self.metrics = {}
        for name, method in inspect.getmembers(self, inspect.ismethod):
            if getattr(method, "ismetric", False):
                self.metrics[name] = method

    def __call__(
        self,
        events: Sequence[dict[str, int | str]],
        metric: str = "most_recent",
        condition: Optional[Callable] = None,
        **kwargs: Any,
    ) -> dict[str, list[int]]:

        if condition is None:
            condition = lambda value: value > self.threshold

        try:
            results = self.metrics[metric](events, **kwargs)
        except KeyError:
            results = self.metrics["most_recent"](events, **kwargs)

        return {key: value for key, value in results.items() if condition(value)}

    def __str__(self) -> str:
        return "MODEL"

    def __repr__(self) -> str:
        return str(self)

    @metric
    def most_recent(self, events: Sequence[dict[str, int | str]]) -> dict[Key, float]:
        event = events[-1]
        key = Key(event["aid"], event["type"])
        try:
            return self.embedding[key]
        except KeyError:
            return {}

    @metric
    def exp_smoothing(
        self, events: Sequence[dict[str, int | str]], alpha: float = 0.1
    ) -> dict[Key, float]:
        ...


def main():
    model = Model(threshold=0)
    events = [
        {"aid": 1254160, "ts": 1661976314324, "type": "clicks"},
        {"aid": 186382, "ts": 1661976400248, "type": "clicks"},
        {"aid": 1254160, "ts": 1661976971766, "type": "clicks"},
        {"aid": 850486, "ts": 1661977190599, "type": "clicks"},
        {"aid": 1254160, "ts": 1661977202848, "type": "clicks"},
        {"aid": 186382, "ts": 1661977231509, "type": "clicks"},
    ]
    print(model(events))


if __name__ == "__main__":
    main()
