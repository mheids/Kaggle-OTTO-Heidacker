'''
Data processing and normalization for OTTO
'''

import random

import numpy as np
import pandas as pd
import pickle

from matplotlib import dates
from IPython.display import display
from tqdm import tqdm
from collections import OrderedDict
import torch
import matplotlib.pyplot as plt
import seaborn as sns

class Embedding:
    def __init__(self) -> None:
        with open('aids.p', 'rb') as f:
            self.aids = pickle.load(f)


if __name__ == '__main__':
    pass