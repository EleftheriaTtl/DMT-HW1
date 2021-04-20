#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:57:04 2021

@author: psh
"""

import pandas as pd
from Shingler import *
import os, time


        

dataset_path = 'part_2/dataset/250K_lyrics_from_MetroLyrics.csv'

df = pd.read_csv(dataset_path)

shingler, table = df_to_shingles(df, 'lyrics')

table.to_csv('dataset_part_2.tsv', index=False, sep="\t")


# ---------------- Running java tool ----------------
n = 300
r = 25
b = 12
j = 0.95

output_file =  f"output/lsh_plus_min_hashing_{j}_{r}_{b}_{n}.tsv"
# cmd = f"cd part_2/part_2_1; java tools.NearDuplicatesDetector lsh_plus_min_hashing \
#         {j} {r} {b} \
#         hash_functions/{n}.tsv \
#         ../../dataset_part_2.tsv \
#         {output_file}"
cmd = f"cd part_2/part_2_1; java tools.NearDuplicatesDetector lsh {r} {b} hash_functions/{n}.tsv ../../dataset_part_2.tsv output/lsh__{r}_{b}_{n}.tsv"
#print(pd.read_csv())


t = time.time()
print(cmd)
os.system(cmd)
print('Timeof NearDuplicate tool:', time.time() - t)
# -----------------------------------------------------


def true_near_duplicates(sets_file, candidates_file, j=0.95):
    sets_df = pd.read_csv(sets_file, sep='\t')
    candidates_df = pd.read_csv(candidates_file, sep='\t')
    pairs = []
    
    for i, row in candidates_df.iterrows():
        s1 = set(sets_df[sets_df['ID'] == row['name_set_1']]['ELEMENTS_IDS'].values[0])
        s2 = set(sets_df[sets_df['ID'] == row['name_set_2']]['ELEMENTS_IDS'].values[0])
        
        real_jaccard = len(s1 & s2) / len(s1 | s2)

        if real_jaccard >= j:
            pairs.append({row['name_set_1'], row['name_set_2']})
            
    return candidates_df, pairs
                

candidates_df, pairs = true_near_duplicates('dataset_part_2.tsv', 
                     "part_2/part_2_1/" + output_file, 
                     j=j)

print('Number of candidates', len(candidates_df))
print('Number of near-duplicates:', len(pairs))

###-----

n = 300
arr = []
for r, b in [(12, 25), (15, 20), (20, 15), (25, 12)]:
    cmd = f"cd part_2/part_2_1; java tools.NearDuplicatesDetector lsh {r} {b} hash_functions/{n}.tsv ../../dataset_part_2.tsv output/lsh__{r}_{b}_{n}.tsv"
    #print(pd.read_csv())
    
    
    t = time.time()
    print(cmd)
    os.system(cmd)
    
    arr.append([r, b, len(pd.read_csv(f"part_2/part_2_1/output/lsh__{r}_{b}_{n}.tsv"))])

print(pd.DataFrame(arr))
