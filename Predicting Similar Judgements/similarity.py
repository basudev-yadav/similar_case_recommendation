import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
from model import Model
import pygraphviz as pgv
import dgl.nn as dglnn
import dgl.function as fn
from train_gsData import *
from test_gsData import *
from hetDotProduct import HeteroDotProductPredictor
from sklearn.metrics import roc_auc_score
from rgcnClass import RGCN

train = train_hetero_graph
test = test_hetero_graph


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    torch.random.manual_seed(1)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

sampling_rate = 1

def manual_auc(scores,labels):
    s = []
    p = []
    for i in range(len(scores)):
        s.append((scores[i])[0])
    l = labels.tolist()
    '''print("SCORES LENGTH: ",len(s))
    print(s)
    print("LABELS LENGTH: ",len(l))
    print(l)'''
    I = 0
    for i in range(len(l)):
        if l[i]==0:
            I = i
            break
    D = I * sampling_rate * I
    np = I
    nn = sampling_rate*I
    counter = 0
    for i in range(np):
        percent = 0
        for j in range(np,np+nn):
            #print("j = ",j)
            if s[i]>s[j]:  
                percent = percent+1
                counter = counter+1
        #print("---------- Positive sample is greater than ",percent/nn," number of negative samples")
        p.append(percent/nn)
        percent = 0
    #print("AUC MANUAL: ",counter/D)
    return p

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    p=manual_auc(scores,labels)
    return roc_auc_score(labels, scores),p

#=====================================================================================

train_neg = construct_negative_graph(train, sampling_rate, ('judgement', 'similarTo', 'judgement'))
test_neg = construct_negative_graph(test,sampling_rate,('judgement', 'similarTo', 'judgement'))
print(train.etypes)
model = Model(27, 10, 2, train.etypes)
judgement_feats = train.nodes['judgement'].data['feature']
test_judgement_feats = test.nodes['judgement'].data['feature']
node_features = {'judgement': judgement_feats}#, 'court': court_feats}
test_node_features = {'judgement': test_judgement_feats}
optimizer = torch.optim.Adam(model.parameters())
pred = HeteroDotProductPredictor()


for epoch in range(600):
    h = model(train,train_neg,node_features,etype='similarTo')
    pos_score = pred(train, h, etype='similarTo')
    #print(len(pos_score))
    neg_score = pred(train_neg, h, etype='similarTo')
    #print(len(neg_score))
    # backward
    optimizer.zero_grad()
    loss = compute_loss(pos_score, neg_score)
    loss.backward()
    optimizer.step()

    print('In epoch {}'.format(epoch))
    
#=============================== FOR TEST ===================================   
print("=========== TEST SET") 

with torch.no_grad():
    pos_score = pred(test, h, etype='similarTo')
    neg_score = pred(test_neg, h, etype='similarTo')
    #node_embeddings = model.sage(test,test_node_features)
    #torch.save(node_embeddings , 'similarity_node_embeddings.pt')
    score,p = compute_auc(pos_score, neg_score)
    print('AUC',score )
    index=[]
    neg_index=[]
    scores=[]
    neg_scores=[]
    for i in range(len(p)):
        if p[i] > sum(p)/len(p):
            index.append(i)
            scores.append(p[i])
        else:
            neg_index.append(i)
            neg_scores.append(p[i])
    #print("INDICES: ",index)
    #print("SCORES: ",scores)
    data = pd.read_csv("input/TEST_SET.csv")
    predicted = pd.DataFrame(columns=['SUBJECT','RELATION','OBJECT','SCORE'])
    not_predicted = pd.DataFrame(columns=['SUBJECT','RELATION','OBJECT','SCORE'])
    print("-------- Predicted Edges ---------")
    for i in range(len(index)):
        #print("-------------- POSITIVE SAMPLES")
        #print(data.loc[index[i],:])
        predicted.loc[i,'SUBJECT'] = data.loc[index[i],'SUBJECT']
        predicted.loc[i,'RELATION'] = data.loc[index[i],'RELATION']
        predicted.loc[i,'OBJECT'] = data.loc[index[i],'OBJECT']
        predicted.loc[i,'SCORE'] = scores[i]
    for i in range(len(neg_index)):
        #print("--------------- NEGATIVE SAMPLES")
        #print(data.loc[neg_index[i],:])
        not_predicted.loc[i,'SUBJECT'] = data.loc[neg_index[i],'SUBJECT']
        not_predicted.loc[i,'RELATION'] = data.loc[neg_index[i],'RELATION']
        not_predicted.loc[i,'OBJECT'] = data.loc[neg_index[i],'OBJECT']
        not_predicted.loc[i,'SCORE'] = neg_scores[i]
    predicted = predicted.drop_duplicates().reset_index(drop=True)
    not_predicted = not_predicted.drop_duplicates().reset_index(drop=True)
    predicted.to_csv("output/predicted_results.csv",index=False)  #remove NEG later !!!!
    not_predicted.to_csv("output/not_predicted_results.csv",index=False)  #remove NEG later !!!

#===================================================================================================

