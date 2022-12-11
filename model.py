# -*- encoding: utf-8 -*-

# Standard Imports
import random
import pickle
import inspect
from pathlib import Path
from functools import partial

# Third Party Imports
import numpy as np
import pandas as pd
from numba import jit, float64, int64
from tqdm import tqdm

# Local import
from train_embedding import Embedding, Key, DefaultFloatDict

# Typing Imports
from typing import Optional, Sequence, Callable, Any

import warnings

warnings.filterwarnings("ignore")


def metric(func: Callable) -> Callable:
    func.ismetric = True
    return func


def make_key(event: dict) -> Key:
    return Key(event["aid"], event["type"])


class EventArray:
    def __init__(
        self, events: Sequence[dict[str, int | str]], embedding: Embedding
    ) -> None:
        self.embedding = embedding
        self.keys = [make_key(event) for event in events]
        self.aids = set()
        for event in events:
            self.aids |= set(self.embedding[make_key(event)])
        self.indexer = {aid: i for i, aid in enumerate(self.aids)}
        self.array = np.array([self.process_event(event) for event in events])

    def __str__(self) -> str:
        return str(self.array)

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self.array)

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape

    def __getitem__(self, item: Key | int) -> np.ndarray:
        if isinstance(item, Key):
            return self.array[:, self.indexer[item]]
        return self.array[item]

    def process_event(self, event: dict[str, int | str]) -> np.ndarray:
        array = np.zeros(len(self.aids))
        results = self.embedding[make_key(event)]
        for key, value in results.items():
            array[self.indexer[key]] = value
        return array

    def to_dict(self) -> dict[Key, float]:
        return {
            event_key: {
                index_key: value
                for index_key, value in zip(self.indexer, self.array[i])
                if value
            }
            for i, event_key in enumerate(self.keys)
        }


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
        try:
            return self.embedding[make_key(event)]
        except KeyError:
            return {}

    @metric
    def ewma(
        self, events: Sequence[dict[str, int | str]], alpha: float = 0.1
    ) -> dict[Key, float]:
        array = EventArray(events, self.embedding)
        data = array.array
        print(data)
        print('\n')
        """Exponential Weighted Moving Average"""

        alpha_rev = 1-alpha
        scale = 1/alpha_rev

        powers = np.ones(data.shape) * np.arange(data.shape[0] - 1, -1, -1)[:, np.newaxis]
        scale_arr = scale**powers
        offset = data*alpha_rev**(powers+1)

        pw0 = alpha*alpha_rev**(powers-1)
        mult = data*pw0*scale_arr
      
        cumsums = mult.cumsum(axis=0)
        out = offset + cumsums*scale_arr[::-1]
        
        out = out[-1]
        print(out)
        print('\n')
        return {key: out[i] for key, i in array.indexer.items()}



def main():
    np.set_printoptions(linewidth=200)

    model = Model(threshold=0)
    events = [
        {"aid": 1254160, "ts": 1661976314324, "type": "clicks"},
        {"aid": 186382, "ts": 1661976400248, "type": "clicks"},
        {"aid": 1254160, "ts": 1661976971766, "type": "clicks"},
        {"aid": 850486, "ts": 1661977190599, "type": "clicks"},
        {"aid": 1254160, "ts": 1661977202848, "type": "clicks"},
        {"aid": 186382, "ts": 1661977231509, "type": "clicks"},
    ]
    output = sorted(list(model(events, "ewma").items()), key = lambda item: item[1], reverse=True)
    for key, value in output:
        print(f"{key}:{' ' * (42-len(str(key)))}{value}")


if __name__ == "__main__":
    main()
