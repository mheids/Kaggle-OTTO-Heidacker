# -*- encoding: utf-8 -*-

# Standard Imports
import random
import pickle
from pathlib import Path

# Third Party Imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local import
from train_embedding import Embedding, Key

# Typing Imports
from typing import Optional, Sequence

import warnings

warnings.filterwarnings("ignore")

class Model:
    def __init__(self, input_path: Optional[str | Path] = None, threshold: float = 0.3):
        self.embedding = Embedding.load(input_path)
        self.threshold = threshold

    def __call__(self, events: Sequence[dict]) -> dict[str, list[int]]:
        key = Key(events[-1]["aid"], events[-1]["type"])

        try:
            results = self.embedding[key]
            return [result for result in results.items() if result[1] >= self.threshold]
        except KeyError:
            return []

def main():
    model = Model(threshold=0)
    events = [{"aid":1254160,"ts":1661976314324,"type":"clicks"},{"aid":186382,"ts":1661976400248,"type":"clicks"},{"aid":1254160,"ts":1661976971766,"type":"clicks"},{"aid":850486,"ts":1661977190599,"type":"clicks"},{"aid":1254160,"ts":1661977202848,"type":"clicks"},{"aid":186382,"ts":1661977231509,"type":"clicks"}]


    print(model(events))

if __name__ == '__main__':
    main() 