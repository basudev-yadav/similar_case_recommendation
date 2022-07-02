import torch
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import csv

def display_ndata(n):
    print("-------- NODE DATA --------")
    print("Length: ",len(n))
    print("Data: ",n)
    print("Type: ",type(n))

def display_edata(e):
    print("-------- EDGE DATA --------")
    print("Length: ",len(e))
    #print("Data: ",e)
    print("Type: ",type(e))
    e = e.tolist()
    print("Data: ",e)

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            h = h['judgement']
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            #print(len(graph.ndata['h']))
            #display_ndata(graph.ndata['h'])
            #print(len(graph.edges[etype].data['score']))
            #display_edata(graph.edges[etype].data['score'])
            return graph.edges[etype].data['score']