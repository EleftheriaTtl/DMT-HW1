# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:09:52 2021

@author: Pavlo, Eleftheria
"""
from __future__ import division, unicode_literals
import codecs

import glob
import time, os, re, math
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from whoosh import index
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.analysis import *
from whoosh.qparser import *
from whoosh import scoring



current_time_msec = lambda: int(round(time.time() * 1000))

class SirIndexer:
    """
    Class for indexing documents
    """

    DOCUMENTS_FOLDER = 'DOCUMENTS/'
    INDEX_FOLDER = 'directory_index_'

    def __init__(self, dataset_path, analyzer, use_title):
        self.directory_containing_the_index = dataset_path + self.INDEX_FOLDER
        self.documents_path = dataset_path + self.DOCUMENTS_FOLDER
        self.use_title = use_title

        # Create schema
        if use_title:
            self.schema = Schema(id=ID(stored=True),
                                title=TEXT(stored=False, analyzer=analyzer),
                                content=TEXT(stored=False, analyzer=analyzer))
        else:
            self.schema = Schema(id=ID(stored=True),
                                content=TEXT(stored=False, analyzer=analyzer))

        # Creating directory for index if it doesn't exist
        if not os.path.exists(self.directory_containing_the_index):
            os.mkdir(self.directory_containing_the_index)

        # Open index
        #self.ix = index.open_dir(self.directory_containing_the_index)

    def create(self):
        """
        Open documents one by one and create index from them

        Returns
        -------
        None.

        """
        # Create an empty-Index with defined schema
        create_in(self.directory_containing_the_index, self.schema)
        ix = index.open_dir(self.directory_containing_the_index)
        
        ts_start = current_time_msec()
        
        writer = ix.writer()
        
        # Get list of documents names
        #arr = os.listdir(self.documents_path)
        arr = glob.glob(self.documents_path + '*.html')

        num_added_records_so_far = 0
        for record in range(len(arr)):
            # Make proper id
            doc_id = re.sub("[^0-9]", "", os.path.basename(arr[record]))
            
            with open(arr[record], 'r') as f:
                # Since we have kind of HTML structure we can use BeautifulSoup to parse fields
                document = BeautifulSoup(f.read(), 'html.parser')
                text = document.body.get_text()
                title = document.title.get_text()
                
                if self.use_title:
                    writer.add_document(id=doc_id, title=title, content=text)  # write according to schema
                else:
                    writer.add_document(id=doc_id, content=text)
    
                num_added_records_so_far += 1
                if (num_added_records_so_far % 100 == 0):
                    print(" num_added_records_so_far= " + str(num_added_records_so_far))


        writer.commit()  # this takes a lot of time, because it's committing all the writing operations

        ts_end = current_time_msec()
        print("TimeStamp: ", time.asctime(time.localtime(time.time())))
        total_time_msec = (ts_end - ts_start)
        print("total_time= " + str(total_time_msec) + "msec")

        self.ix = ix

class SirSearchEngine:
    """
    Class to perform search
    """
    #ROUND_SCORE = 5
    
    def __init__(self, indexer, scoring_func, max_number_of_results=5, use_title=True):
        self.max_number_of_results = max_number_of_results
        self.ix = indexer.ix
        
        # Initialization of query parser depending on number of fields
        self.query_parser = MultifieldParser(["title", "content"], self.ix.schema) if use_title else QueryParser("content", self.ix.schema)
        
        self.searcher = self.ix.searcher(weighting=scoring_func)

    def results(self, q):
        """
        Finds results of q

        Parameters
        ----------
        q : query

        Returns
        -------
        Results with scores 

        """
        parsed_query = self.query_parser.parse(q)
        return self.searcher.search(parsed_query, limit=self.max_number_of_results)

    def close(self):
        self.searcher.close()
        

    def multi_results_from_df(self, query_df):
        """
        Finds results for each query in input dataframe of quries

        Parameters
        ----------
        query_df : dataframe of queries

        Returns
        -------
        res_df : dataframe with results where found documents grouped in a list

        """
        arr = []
        
        # Creating list of results for each query. Each entry in 'arr' is one document for one query
        for i in range(len(query_df)):
            results = self.results(query_df['Query'][i])

            for hit in results:
                arr.append([query_df['Query_id'][i], hit['id'], hit.rank + 1, hit.score]) # hit['id'] - document id
        
        
        # Grouping results and combaning them in list
        res_df = pd.DataFrame(arr, columns=('Query_ID', 'Doc_ID', 'Rank', 'Score'))\
                    .astype({'Doc_ID': 'int'})\
                    .groupby('Query_ID')['Doc_ID'].apply(list)\
                    .reset_index()
                
        return res_df


def MRR(Query_Total,SE_list):
  somma = 0
  for query_id in Query_Total['Query_id']:
      se_results = SE_list[SE_list['Query_ID'] == query_id]['Doc_ID'].values[0]
      gt_results = Query_Total[Query_Total['Query_id'] == query_id]['Relevant_Doc_id'].values[0]
      
      if set(gt_results) & set(se_results) == set():
          continue
      else:
          index_q = min([idx + 1 for idx, val in enumerate(SE_list[SE_list['Query_ID'] == query_id]['Doc_ID'].values[0]) if int(val) in set(gt_results) & set(se_results)])        
          somma = somma +  1/index_q
  return somma/len(Query_Total)

def R_Precision(query_id, Query_Total, SE_list):
    se_results = SE_list[SE_list['Query_ID'] == query_id]['Doc_ID'].values[0]
    gt_results = Query_Total[Query_Total['Query_id'] == query_id]['Relevant_Doc_id'].values[0]
    return len(list(set(gt_results) & set(se_results[:len(gt_results)])))/len(gt_results)

def R_Precision_dist(Query_Total, SE_list):
    R_P = {}
    for query_id in Query_Total['Query_id']:
        R_P[query_id] = R_Precision(query_id, Query_Total, SE_list)
        
    mean_R_P = np.mean(list(R_P.values()))
    min_R_P = np.min(list(R_P.values()))
    max_R_P = np.max(list(R_P.values()))
    median_R_P = np.median(list(R_P.values()))
    qua_1_R_P = np.percentile(list(R_P.values()),25)
    qua_3_R_P = np.percentile(list(R_P.values()),75)
    
    return mean_R_P, min_R_P, max_R_P, median_R_P, qua_1_R_P, qua_3_R_P

def P_at_k(query_id, k, Query_Total, SE_list):
    se_results = list(map(int, SE_list[SE_list['Query_ID'] == query_id]['Doc_ID'].values[0]))
    gt_results = Query_Total[Query_Total['Query_id'] == query_id]['Relevant_Doc_id'].values[0]
    return len(list(set(gt_results) & set(se_results[:k])))/min(k, len(gt_results))


def P_at_k_plot(Query_Total, SE_list, label):
    p_a_k = {}
    ks = [1, 3, 5, 10]
    for k in ks:
        p_a_k[k] = sum(P_at_k(query_id, k, Query_Total, SE_list) for query_id in Query_Total['Query_id'])/len(Query_Total)
        
    plt.plot(ks, p_a_k.values(), label=label)
    
def nDCG(query_id, k, Query_Total, SE_list):
    se_results = list(map(int, SE_list[SE_list['Query_ID'] == query_id]['Doc_ID'].values[0]))
    gt_results = Query_Total[Query_Total['Query_id'] == query_id]['Relevant_Doc_id'].values[0]
    #
    idcg = 0
    dcg = 0
    for p in range(1,k+1):
      dcg = dcg + (se_results[p-1] in gt_results)/math.log(p+1,2)
      # We have that IDCG is the performance of an ideal search engine
      # So whenever we take into account k > |GT(q)|, we'll get IDCG(k, q) = IDCG(|GT(q),q)
      # since the relevenant documents (for k > |GT(q)|), would have alreay been taken into account
      # All the others would get relevance = 0
      if p <= len(gt_results):
          idcg = idcg + 1/math.log(p+1,2)
    return dcg/idcg

def nDCG_plot(Query_Total, SE_list, label):
    ndcg = {}
    ks = [1, 3, 5, 10]
    
    for k in ks:
        ndcg[k] = sum(nDCG(query_id, k, Query_Total, SE_list) for query_id in Query_Total['Query_id'])/len(Query_Total)

    
    plt.plot(ks, ndcg.values(), label=label)


def run_configuration(dataset_path, analyzer, scroring_func, max_number_of_results, Query_Total, use_title):   
    """
    Running configuration for particular analyzer and scoring function

    """
    # Indexing
    sir_indexer = SirIndexer(dataset_path, analyzer, use_title=use_title)
    sir_indexer.create()
    
    # Search Engine
    sir_se = SirSearchEngine(sir_indexer, scroring_func, max_number_of_results=max_number_of_results, use_title=use_title)
    SE_list = sir_se.multi_results_from_df(Query_Total)
    sir_se.close()
    
    return {
            'MRR': MRR(Query_Total, SE_list), 
            'Dist': R_Precision_dist(Query_Total, SE_list),
            'SE_list': SE_list
            }




def full_analysis(prefix, dataset_name, use_title):
    files_path = 'part_1/part_1_1/'
    dataset_path = files_path + dataset_name + '/'
    #prefix = re.sub("_.+", "", dataset_name).lower()
    
    GT = pd.read_csv(dataset_path + prefix + '_Ground_Truth.tsv', sep='\t')
    GT_list = GT.groupby('Query_id')['Relevant_Doc_id'].apply(list)
    
    queries = pd.read_csv(dataset_path + prefix + '_Queries.tsv', sep='\t')
    queries.rename(columns = {'Query_ID' : 'Query_id'}, inplace = True)
    Query_Total = queries.merge(GT_list, how='inner', on='Query_id')
    
    MRR_df = []
    R_prec_df = []
    conf_df = []
    
    analyzers = [('simple', SimpleAnalyzer()), ('standard', StandardAnalyzer()), 
                     ('stemm', StemmingAnalyzer()), ('fancy', FancyAnalyzer())]
    
    scoring_funcs = [('tf_idf', scoring.TF_IDF()), ('BM25F', scoring.BM25F()),('Freq', scoring.Frequency())]
    
    
    # Finding max number of ground truth - this will be max number 
    # of results that we have to obtain from SE
    max_number_of_results = GT.groupby('Query_id').count().max()[0]
    conf_db = {}
    i = 1
    for a_name, analyzer in analyzers:
        for s_name, scoring_func in scoring_funcs:
            if s_name == 'Freq' and a_name in ['simple', 'standard', 'fancy']:
                continue
            conf_info = run_configuration(dataset_path, analyzer, scoring_func,max_number_of_results, Query_Total, use_title=True)
            conf_name = a_name + '_' + s_name
            search_ = 'SE_'+str(i)
            conf_db[search_] = conf_info
            conf_df.append([search_, conf_name])
            MRR_df.append([search_, conf_info['MRR']])
            R_prec_df.append([search_] + list(conf_info['Dist']))
            i = i+1
    conf_df = pd.DataFrame(conf_df, columns=['Search Engine', 'Configuration'])
    MRR_df = pd.DataFrame(MRR_df, columns=['Name', 'MRR'])
    R_prec_df = pd.DataFrame(R_prec_df, columns=['Conf.', 'Mean', 'Min', 'Max', 'Median', 'Q1', 'Q2'])
    
    
    MRR_df = MRR_df.sort_values(by='MRR', ascending=False)
    top_5 = MRR_df.head(5)
    #print(top_5)
    # Plotting
    
    for name in top_5['Name']:
        P_at_k_plot(Query_Total, conf_db[name]['SE_list'], name )
    plt.xlabel('k')
    plt.ylabel('P@k')
    plt.ylim(0, 0.75)
    plt.legend()
    plt.figure()
    
    for name in top_5['Name']:
       nDCG_plot(Query_Total, conf_db[name]['SE_list'], name)
    plt.xlabel('k')
    plt.ylabel('nDCG@k')
    plt.ylim(0, 0.75)
    plt.legend()
    plt.figure()
    
    return conf_df, MRR_df, top_5, R_prec_df
    



# Cranfield DATASET
c_conf_df, c_MRR_df, c_top_5, c_R_prec_df = full_analysis('cran', dataset_name = 'Cranfield_DATASET', use_title=True)

# Time DATASET
#t_MRR_df, conf_df, t_MRR_df, t_top_5, t_R_prec_df = full_analysis('time', dataset_name = 'Time_DATASET', use_title=False)


