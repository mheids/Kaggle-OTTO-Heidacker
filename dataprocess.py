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

chunks = pd.read_json('./Data/train.jsonl', lines=True, chunksize=100_000)

train = pd.DataFrame()

for i, chunk in tqdm(enumerate(chunks)):
    event_dict = {
        'customer_id': [],
        'product_code':     [],
        'time_stamp':      [],
        'event_type':    []
    }

    if i >= 2:
        break
    for session, events in zip(chunk['session'].tolist(), \
                               chunk['events'].tolist()):
        for event in events:
            event_dict['customer_id'].append(session)
            event_dict['product_code'].append(event['aid'])
            event_dict['time_stamp'].append(event['ts'])
            event_dict['event_type'].append(event['type'])
    chunk_session = pd.DataFrame(event_dict)
    train = pd.concat([train, chunk_session])

train = train.reset_index(drop=True)

train['time_stamp'] = pd.to_datetime(train['time_stamp'], unit='ms')

train.loc[:5000].to_csv('traindata.csv')