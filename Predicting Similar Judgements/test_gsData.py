import dgl
import dgl.nn
import numpy as np 
import pandas as pd 
import json
import csv
import torch
import re
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
print(dgl.nn.__file__)

#================================================================================================================

def return_edges():
    data = pd.read_csv("input/PREDICT_TEST_SET.csv") # remove NEG later !!!
    src = data.loc[:,'SUBJECT']
    src = src.values
    dest = data.loc[:,'OBJECT']
    dest = dest.values
    return src,dest

def return_features_labels():
    data = pd.read_csv("input/BERT_FEATURES_AVERAGE.csv") #labels.csv
    
    
  

    with open('input/MAPPING.json') as json_file: 
        j = json.load(json_file)
    for i in range(len(data)):
        data.loc[i,'tid'] = j[str(int(data.loc[i,'tid']))]  #fname or tid
    lofl=[]
    for i in range(len(data)):
        l = data.loc[i,:].values.tolist()
        l = l[1:]
        lofl.append(l)
    return torch.FloatTensor(lofl)

#================================================================================================================

with open('input/MAPPING.json') as json_file: 
    judgements = json.load(json_file)

n_judgements = len(judgements)

judgements_src , judgements_dest = return_edges()

test_hetero_graph = dgl.heterograph({

    ('judgement','similarTo','judgement') : (judgements_src,judgements_dest)})

features = return_features_labels()

test_hetero_graph.nodes['judgement'].data['feature'] = features

