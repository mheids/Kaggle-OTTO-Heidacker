# -*- encoding: utf-8 -*-

"""
Data processing and normalization for OTTO
"""

# Standard Imports
import random

# Third Party Imports
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    chunks = pd.read_json("./Data/train.jsonl", lines=True, chunksize=100_000)

    aids = set()

    for i, chunk in tqdm(enumerate(chunks)):
        for events in chunk["events"]:
            for event in events:
                aids.add(event["aid"])

    with open("aids.p", "wb+") as f:
        pickle.dump(aids, f)
