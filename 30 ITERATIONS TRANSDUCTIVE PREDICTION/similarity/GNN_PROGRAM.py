#!/usr/bin/env python

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
from model import Model
#import pygraphviz as pgv
import dgl.nn as dglnn
import dgl.function as fn
from train_gsData import *
from test_gsData import *
from hetDotProduct import HeteroDotProductPredictor
from sklearn.metrics import roc_auc_score
from rgcnClass import RGCN


sampling_rate = 1

global feature


res_text = open("res.txt","w")


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


def manual_auc(scores, labels):
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
        if l[i] == 0:
            I = i
            break
    D = I * sampling_rate * I
    np = I
    nn = sampling_rate * I
    counter = 0
    for i in range(np):
        percent = 0
        for j in range(np, np + nn):
            # print("j = ",j)
            if s[i] > s[j]:
                percent = percent + 1
                counter = counter + 1
        # print("---------- Positive sample is greater than ",percent/nn," number of negative samples")
        p.append(percent / nn)
        percent = 0
    # print("AUC MANUAL: ",counter/D)
    return p


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    p = manual_auc(scores, labels)
    return roc_auc_score(labels, scores), p


score = 0

#for j in ["NER_LAW","LAWPOINTS","DOMAIN_EXPERT","BERT"]:
for j in ["BERT"]:
    feature = j

    ft = open("feat.txt","w")
    print("writing!!!")
    ft.write(str(feature))
    ft.close()
    for epochs in [40,60]:
        train = train_hetero_graph
        test = test_hetero_graph

        input_layer_nodes = 0
        hidden_layer_nodes = 0
        if feature == "NER_LAW":
            input_layer_nodes = 28
            hidden_layer_nodes = 18
        elif feature == "LAWPOINTS":
            input_layer_nodes = 14
            hidden_layer_nodes = 9
        elif feature == "DOMAIN_EXPERT":
            input_layer_nodes = 85
            hidden_layer_nodes = 56
        else:
            input_layer_nodes = 768
            hidden_layer_nodes = 520


        train_neg = construct_negative_graph(train, sampling_rate, ('judgement', 'similarTo', 'judgement'))
        test_neg = construct_negative_graph(test, sampling_rate, ('judgement', 'similarTo', 'judgement'))
        print(train.etypes)
        model = Model(input_layer_nodes, hidden_layer_nodes, 2, train.etypes)
        judgement_feats = train.nodes['judgement'].data['feature']
        test_judgement_feats = test.nodes['judgement'].data['feature']
        node_features = {'judgement': judgement_feats}  # , 'court': court_feats}
        test_node_features = {'judgement': test_judgement_feats}
        optimizer = torch.optim.Adam(model.parameters())
        pred = HeteroDotProductPredictor()

        for epoch in range(epochs):
            h = model(train, train_neg, node_features, etype='similarTo')
            pos_score = pred(train, h, etype='similarTo')
            # print(len(pos_score))
            neg_score = pred(train_neg, h, etype='similarTo')
            # print(len(neg_score))
            # backward
            optimizer.zero_grad()
            loss = compute_loss(pos_score, neg_score)
            loss.backward()
            optimizer.step()

            print('In epoch {}'.format(epoch))

        with torch.no_grad():
            pos_score = pred(test, h, etype='similarTo')
            neg_score = pred(test_neg, h, etype='similarTo')
            # node_embeddings = model.sage(test,test_node_features)
            # torch.save(node_embeddings , 'similarity_node_embeddings.pt')
            score, p = compute_auc(pos_score, neg_score)
            # score contains the AUC of predefined function
            # p contains the AUC computed by manual AUC
            print('AUC', score)
            index = []
            neg_index = []
            scores = []
            neg_scores = []
            for i in range(len(p)):
                if p[i] > sum(p) / len(p):
                    index.append(i)
                    scores.append(p[i])
                else:
                    neg_index.append(i)
                    neg_scores.append(p[i])
            # print("INDICES: ",index)
            # print("SCORES: ",scores)
            data = pd.read_csv("input/TEST_SET_SAMPLED.csv")
            predicted = pd.DataFrame(columns=['SUBJECT', 'RELATION', 'OBJECT', 'SCORE'])
            not_predicted = pd.DataFrame(columns=['SUBJECT', 'RELATION', 'OBJECT', 'SCORE'])
            print("-------- Predicted Edges ---------")
            for i in range(len(index)):
                print("-------------- POSITIVE SAMPLES")
                print(data.loc[index[i], :])
                predicted.loc[i, 'SUBJECT'] = data.loc[index[i], 'SUBJECT']
                predicted.loc[i, 'RELATION'] = data.loc[index[i], 'RELATION']
                predicted.loc[i, 'OBJECT'] = data.loc[index[i], 'OBJECT']
                predicted.loc[i, 'SCORE'] = scores[i]
            for i in range(len(neg_index)):
                print("--------------- NEGATIVE SAMPLES")
                print(data.loc[neg_index[i], :])
                not_predicted.loc[i, 'SUBJECT'] = data.loc[neg_index[i], 'SUBJECT']
                not_predicted.loc[i, 'RELATION'] = data.loc[neg_index[i], 'RELATION']
                not_predicted.loc[i, 'OBJECT'] = data.loc[neg_index[i], 'OBJECT']
                not_predicted.loc[i, 'SCORE'] = neg_scores[i]
            predicted = predicted.drop_duplicates().reset_index(drop=True)
            not_predicted = not_predicted.drop_duplicates().reset_index(drop=True)
            predicted.to_csv("output/predicted_results.csv", index=False)  # remove NEG later !!!!
            not_predicted.to_csv("output/not_predicted_results.csv", index=False)  # remove NEG later !!!

            res_text.write(feature)
            res_text.write("\t")
            res_text.write(epochs)
            res_text.write("\t")
            res_text.write(sampling_rate)
            res_text.write("\t")
            res_text.write(score)
            res_text.write("\n")
            res_text.close()

# Flow of the program
# 
# Input: 
# 1. OE_LABELS(csv file) : contains features of the nodes
# 2. MAPPING.json : maps the document ids to node ids
# 3. TRAIN_SET(csv) : contains data in triples format where each row indicates an edge (docid,similarTo,docid)
# 4. TEST_SET(csv) : contains data in triples format where each row indicates an edge (docid,similarTo,docid)
# 
# Flow(as understood)
# 1. Create a heterograph of train set
# 2. Create a heterograph of test set
# 3. Perform negative sampling in both the graphs
# 4. Initialize RGCN model and pass parameters
# 5. Train the model on using the train graph with negative edges as well
# 6. Test the model using the TEST_SET
# 7. Compute AUC score
# 8. Compute the probabilities of similarity between two nodes in an edge
# 9. Parse this probability array
#     if prob. is greater than average(prob.)
#         keep in predicted results
#     else
#         keep in non predicted results

# In[ ]:




