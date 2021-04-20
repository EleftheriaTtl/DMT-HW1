#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:37:58 2021

@author: psh
"""

import whoosh as wh
from whoosh import analysis
import pandas as pd


class Shingler:
    """ 
    Class for making vocabulary of shingles 
    """
    
    def __init__(self, size):
        self.vocab = dict()
        self.size = size # Number of shingles
        self.counter = 0
        
        self.analyzer = wh.analysis.SimpleAnalyzer()
    
    def add_and_return(self, text):
        """
        Adds shingles to vocabulary and return their ids 

        Parameters
        ----------
        text : string

        Returns
        -------
        res : list of ids of shingles

        """
        # We need to maintain set of shingles
        res = set()
        
        for sh in self.shingle(text):
            # Check if shingle is already in vocabulary
            if not sh in self.vocab:
                # Adding new shingle to vocabulary with id equal to current numnber of shingles
                self.vocab[sh] = self.counter
                self.counter += 1
                
            res.add(self.vocab[sh])
            
        return res
    
    def shingle(self, text):
        """
        Create shingles of provided text

        Parameters
        ----------
        text : string

        Yields
        ------
        shingle : current shingle

        """
        # Using whoosh analyzer to make tokens and convert them to list
        words = [token.text for token in self.analyzer(text)]
        
        assert len(words) > 0
        
        # if number of words less than size we consider it as one shingle
        if len(words) < self.size:
            print(text)
            yield tuple(words)
        else:
            for i in range(len(words) - self.size + 1):
                # Shingle is tuple for "n" words where n is self.size
                shingle = tuple(words[i + k] for k in range(self.size))
                
                yield shingle
        
        
def df_to_shingles(df, column_for_shingle, size=3, before_shingle=None):
    """
    Take dataframe and traverse it by rows to convert specified column values to shingles

    Parameters
    ----------
    df : pandas Data Frame
    column_for_shingle : text column to be shingled
    size : size of shingles for Shingler class
    before_shingle : function, if exists, is executed on text variable before 
    passing to Shingler. It is used in part 2.2 to remove "-" and punctuation

    Returns
    -------
    shingler object, new data frame with columns: ID, ELEMENTS_IDS - set of shingles

    """
    shingler = Shingler(size)
    table = []
    
    for index, row in df[['ID', column_for_shingle]].iterrows():
        ID = row['ID']
        
        # Go to next row if title is not string (somethimes it's nan)
        if not type(row[column_for_shingle]) == str:
            continue
        
        text_to_shingle = row[column_for_shingle] if before_shingle is None else before_shingle(row[column_for_shingle])
        shingles = shingler.add_and_return(text_to_shingle)

        # Convert set of shingles to list and then to str so we have square breakts
        table.append([ID, str(list(shingles))])
        
    return shingler, pd.DataFrame(table, columns=("ID", "ELEMENTS_IDS"))