# Common block
import pandas as pd 
import numpy as np
import random 

files_path = 'part_1/part_1_2/'
dataset_name = 'dataset'
dataset_path = files_path + dataset_name + '/'

# Reading  

GT = pd.read_csv(dataset_path + 'part_1_2__Ground_Truth.tsv', sep='\t')
GT_list = pd.DataFrame(GT.groupby('Query_ID')['Relevant_Doc_id'].apply(list))
GT_list = GT_list.merge(GT.groupby('Query_ID').count().rename(columns={'Relevant_Doc_id': 'Length'}), on='Query_ID')
GT_list = GT_list.reset_index()

SE1 = pd.read_csv(dataset_path + 'part_1_2__Results_SE_1.tsv', sep='\t')
SE1 = SE1[np.isin(SE1['Rank'], [1,2,3,4])]
SE1_list = SE1.groupby('Query_ID')['Doc_ID'].apply(list)
SE1_list = SE1_list.reset_index()

SE2 = pd.read_csv(dataset_path + 'part_1_2__Results_SE_2.tsv', sep='\t')
SE2 = SE2[np.isin(SE2['Rank'], [1,2,3,4])]
SE2_list = SE2.groupby('Query_ID')['Doc_ID'].apply(list)
SE2_list = SE2_list.reset_index()

SE3 = pd.read_csv(dataset_path + 'part_1_2__Results_SE_3.tsv', sep='\t')
SE3 = SE3[np.isin(SE3['Rank'], [1,2,3,4])]
SE3_list = SE3.groupby('Query_ID')['Doc_ID'].apply(list)
SE3_list = SE3_list.reset_index()


def R_Precision(query_id, Query_Total, SE_list):
    se_results = SE_list[SE_list['Query_ID'] == query_id]['Doc_ID'].values[0]
    gt_results = Query_Total[Query_Total['Query_ID'] == query_id]['Relevant_Doc_id'].values[0]
    
    return len(list(set(gt_results) & set(se_results[:len(gt_results)])))/len(gt_results)  

def P_at_k(query_id, k, Query_Total, SE_list):
    se_results = list(map(int, SE_list[SE_list['Query_ID'] == query_id]['Doc_ID'].values[0]))
    gt_results = Query_Total[Query_Total['Query_ID'] == query_id]['Relevant_Doc_id'].values[0]
    return len(set(gt_results) & set(se_results[:k]))/min(k, len(gt_results))


def precision_recall(GT_list, SE1_list):
    '''
    Calculate the average precision, recall, f1 score, precision@k 
    and R- precision for a given SE
    '''
    prec = 0
    reca = 0
    f1 = 0
    r_precision = 0
    p_at_k = 0
    num_queries = len(GT_list['Query_ID'])
    
    for query_id in GT_list['Query_ID']:

        se_results = SE1_list[SE1_list['Query_ID'] == query_id]['Doc_ID'].values[0]
        gt_results = GT_list[GT_list['Query_ID'] == query_id]['Relevant_Doc_id'].values[0]
        
        # precision
        pr = len(list(set(gt_results) & set(se_results)))/4
        
        # recall 
        recall = len((set(gt_results) & set(se_results)))/len(gt_results)
        
        prec = prec + pr
        reca = reca + recall
        
        r_precision += R_Precision(query_id, GT_list, SE1_list)
        p_at_k += P_at_k(query_id, 4, GT_list, SE1_list)
        
        if (pr + recall)!=0:
            f1 = f1 + (2*pr*recall)/(pr + recall)
    
    return p_at_k/num_queries, r_precision/num_queries, prec/num_queries, reca/num_queries, f1/num_queries
            




dfs = []
for len_filter in [GT_list['Length'] > 0]:
    filtered_ids = GT_list[len_filter]['Query_ID']
    res = []
    for sel in [SE1_list, SE2_list, SE3_list]:
        sel = sel[sel['Query_ID'].isin(filtered_ids)]
        p_at_k, r_precision, p, r, f_in_loop = precision_recall(GT_list[len_filter], sel)
        
        # here we calculate the f1 score from the average precision and average recall
        f_outside_loop = (2*p*r)/(p + r)
        # here we calculate the f1 score given to us by R-precision and recall
        f_r_prec_recall = (2*r_precision*r)/(r_precision + r)
        #here we calculate the f1 score given to us by P@k and recall
        f1_at_k = (2*p_at_k*r)/(p_at_k + r)
        
        res.append([p_at_k, p, r, f_outside_loop, f1_at_k, r_precision, f_r_prec_recall])
    
    dfs.append(pd.DataFrame(res, columns=("P_at_K", "Precision", "Recall", "F1", "F1_at_K", "R-precision", "FR_1")))
    

print(dfs[0][['P_at_K', 'Recall', 'F1_at_K']])
print()


