# -*- encoding: utf-8 -*-

"""
Data processing and normalization for OTTO
"""

# Standard Imports
from __future__ import annotations
import pickle
import math
from enum import Enum
from collections import defaultdict, namedtuple
from pathlib import Path
from time import perf_counter

# Third Party Imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit

# Typing Imports
from typing import Optional, Sequence, Any


def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def rawgencount(filename):
    f = open(filename, "rb")
    f_gen = _make_gen(f.raw.read)
    return sum(buf.count(b"\n") for buf in f_gen)


Key = namedtuple("Key", ["aid", "action"])


class DefaultFloatDict:
    def __new__(cls):
        return defaultdict(float)


class Action(Enum):
    CLICKS = "clicks"
    CARTS = "carts"
    ORDERS = "orders"


class Embedding:
    def __init__(self, window: tuple[int, int] = (0, 1), verbose: bool = True) -> None:
        self.values = defaultdict(DefaultFloatDict)
        self.window = window
        self.finalized = False
        self.verbose = verbose

    def __getstate__(self) -> dict:
        self.finalize()
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        for k, v in state.items():
            self.__dict__[k] = v

    def __getitem__(self, item: Any) -> dict:
        return self.values[item]

    def save(
        self, output: Optional[str | Path] = None, verbose: bool | None = None
    ) -> None:
        output = Path("./embeddings/latest.p").resolve() if output is None else output
        verbose = self.verbose if verbose is None else verbose
        if verbose:
            print("Saving Embeddings...")
        with open(output, "wb+") as f:
            pickle.dump(self, f)
        return self

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> Embedding:
        path = Path("./embeddings/latest.p").resolve() if path is None else path
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_window(self, events: Sequence, index: int) -> list[dict]:
        lookback, lookahead = self.window
        min_bound = 0 if lookback is None else max(index - lookback, 0)
        max_bound = (
            len(events) + 1
            if lookback is None
            else min(index + lookahead, len(events)) + 1
        )
        before = events[min_bound:index]
        after = events[index + 1 : max_bound]
        return [(event, 1) for event in before + after]

    def train_chunk(self, data: pd.DataFrame, verbose: bool | None = None) -> None:
        verbose = self.verbose if verbose is None else verbose
        for session in tqdm(data.iterrows(), total=len(data), disable=not verbose):
            events = session[1]["events"]
            for i, event in enumerate(events):
                key1 = Key(event["aid"], event["type"])
                for other_event, value in self.get_window(events, i):
                    key2 = Key(other_event["aid"], other_event["type"])
                    self.values[key1][key2] += value

    def train(
        self,
        n_chunks: Optional[int] = None,
        chunksize: int = 10_000,
        data_path: Optional[Path | str] = None,
        verbose: bool | None = None,
    ) -> None:
        verbose = self.verbose if verbose is None else verbose
        data_path = (
            Path("./data/train.jsonl").resolve() if data_path is None else data_path
        )
        num_lines = 12_899_779
        if n_chunks is None:
            total = num_lines
            n_chunks = (num_lines // chunksize) + 1
        else:
            total = min(n_chunks * chunksize, num_lines)

        chunks = pd.read_json(data_path, lines=True, chunksize=chunksize)

        with tqdm(
            total=total, disable=not verbose, desc="Training Embeddings", smoothing=0
        ) as pbar:
            for i, chunk in enumerate(chunks):
                if i >= n_chunks:
                    break
                self.train_chunk(chunk, verbose=False)
                pbar.update(len(chunk))
        return self

    def finalize(self, verbose: bool | None = None) -> None:
        verbose = self.verbose if verbose is None else verbose
        if not self.finalized:
            if verbose:
                print("Normalizing Embeddings...")
            nums = np.array(
                [value for values in self.values.values() for value in values.values()]
            )
            max_nums = max(nums)
            nums /= max_nums
            logs = np.log(nums)

            min_adjust = min(logs) - (1 / math.log(max_nums))
            max_value = max(logs + abs(min_adjust))

            adjust_value = (
                lambda value: (math.log(value / max_nums) + abs(min_adjust)) / max_value
            )

            for k1, subdict in tqdm(
                self.values.items(), disable=not verbose, desc="Finalizing Embeddings"
            ):
                for k2, value in subdict.items():
                    self.values[k1][k2] = adjust_value(value)

            self.finalized = True
        return self

    def pretty_print(self, innerlimit=5, outerlimit=5) -> None:
        format_key = lambda key: f"AID {key.aid} | {key.action}"
        total = 0
        stop = False
        for i, (k, v) in enumerate(self.values.items()):
            total += len(v)

            if not stop:
                print(format_key(k))
                for j, (k2, v2) in enumerate(v.items()):
                    print(f"    {format_key(k2)} : {v2}")
                    if j >= innerlimit - 1:
                        print(f"    {len(v) - innerlimit} other relationships...")
                        break
                print()
                if i >= outerlimit - 1:
                    stop = True
        print(f"{total} total relationships")
        return self


def train_set(embeddings: dict[str, tuple[int, int]], verbose: bool = True) -> None:
    for name, window in embeddings.items():
        out_path = Path(f"./embeddings/{name}.p").resolve()
        Embedding(window, verbose).train().save(out_path)


def train_from_file(file: str | Path, verbose: bool = True) -> None:
    ...


def main():
    start = perf_counter()
    embedding = Embedding(verbose=True).train(10)
    print(f"Trained embeddings in {perf_counter() - start:.3f} seconds")
    embedding.save()

    start = perf_counter()
    Embedding.load().pretty_print()
    print(f"Loaded embeddings in {perf_counter() - start:.3f} seconds")


if __name__ == "__main__":
    main()
