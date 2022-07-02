import torch
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import csv
from hetDotProduct import HeteroDotProductPredictor
from rgcnClass import RGCN

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()
    def forward(self, g, neg_g, x, etype):
        h = self.rgcn(g, x)
        return h