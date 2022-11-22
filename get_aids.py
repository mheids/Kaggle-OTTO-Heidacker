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
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    chunks = pd.read_json('./Data/train.jsonl', lines=True, chunksize=100_000)

    aids = set()

    for i, chunk in tqdm(enumerate(chunks)):
        for session, events in zip(chunk['session'].tolist(), \
            chunk['events'].tolist()):

            for event in events:
                aids.add(event['aid'])

    with open('aids.p', 'wb+') as f:
        pickle.dump(aids, f)