import torch
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import csv
import dgl
import collections.abc

# Define a Heterograph Conv model

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
    
    def __contains__(self, val):
        return (val == self).any().item()

    def forward(self, graph, inputs):
        # inputs are features of nodes
        #print(type(graph))
        #print(type(inputs))
        #print(isinstance(torch.Tensor(inputs),collections.abc.Sequence))
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h