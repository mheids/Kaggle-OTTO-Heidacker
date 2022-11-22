# -*- encoding: utf-8 -*-

"""
Data processing and normalization for OTTO
"""

# Standard Imports
import random
import pickle
from enum import Enum

# Third Party Imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


class Actions(Enum):
    CLICKS = "clicks"
    CARTS = "carts"
    ORDERS = "orders"


class ActionIndexer:
    def __init__(self):
        self.indexes = {
            Actions.CLICKS: {
                Actions.CLICKS: 0,
                Actions.CARTS: 1,
                Actions.ORDERS: 2,
            },
            Actions.CARTS: {
                Actions.CLICKS: 3,
                Actions.CARTS: 4,
                Actions.ORDERS: 5,
            },
            Actions.ORDERS: {
                Actions.CLICKS: 6,
                Actions.CARTS: 7,
                Actions.ORDERS: 8,
            },
        }

    def __getitem__(self, actions):
        action1, action2 = actions
        return self.indexes[action1][action2]


class Embedding:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open("aids.p", "rb") as f:
            self.aids = pickle.load(f)
        self.n_aids = len(self.aids)
        self.embedding = torch.zeros((self.n_aids, self.n_aids, 9))


if __name__ == "__main__":
    print(Embedding().embedding.shape())
