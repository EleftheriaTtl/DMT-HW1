#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 12:49:03 2021

@author: psh
"""

import pandas as pd
from Shingler import *
import re
from collections import defaultdict
from itertools import  combinations
import time


def make_proper_title(text):
    # Replacing punctuation and "-" by space    
    # I think it's better to replace with space since there might be 
    # abbriviations with dots like B.I.G. 
    return re.sub("[\-.,?!:]", " ", text)


dataset_path = 'part_2/dataset/250K_lyrics_from_MetroLyrics.csv'

df = pd.read_csv(dataset_path)

t1 = time.time()

# Make table of shingle sets
shingler, table = df_to_shingles(df, 'song', before_shingle=make_proper_title)

t2 = time.time()

# Dictionary with keys as set of shingles and elements of list
buckets = defaultdict(list)

for i, row in table.iterrows():
    # Make hashble set of shingles
    key = frozenset(eval(row['ELEMENTS_IDS']))
    
    # Add in a bucket song id
    buckets[key].append(row['ID'])
    

pairs = []
for key in buckets:
    if len(buckets[key]) > 1:
        # Make list of combinations and add it to full list of pairs
        pairs += list(combinations(buckets[key], 2))
        
print('Time with shingles: %s; time of algo: %s' % (time.time() - t1, time.time() - t2))