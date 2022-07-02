#!/usr/bin/env python
# coding: utf-8

# ## Todo
# 
# - Adjacency Matrix (A_sp.npy)
# - Features Matrix (X_sp.npy)
# - nodes_keep and nodes_hide
# - A_train and X_train
# - A_sp -> test_ones and test_zeros
# - A_train -> val_ones and val_zeros
# - Train_ones (edges in A_train)

# In[28]:


path = "../data/CiteSeer/"



print("File prepare data called!!!!!!!!!")
# In[29]:


features = open("FEATURES1.txt","r")


# In[30]:


import pandas as pd
import numpy as np
import json
import re
import os


# In[31]:


train = pd.read_csv("TRAIN_SET.csv")
test = pd.read_csv("TEST_SET.csv")


# In[32]:


T = pd.concat([train,test])


# In[33]:


T = T.drop_duplicates(keep='first')


# In[34]:


#unique nodes in T
nodes = []
for i in range(T.shape[0]):
    nodes.append(int(T.iloc[i][0]))
    nodes.append(int(T.iloc[i][2]))

nodes = list(set(nodes))


# In[35]:


feature_names = ["w_{}".format(ii) for ii in range(86)]


# In[36]:


h = ['id']


# In[37]:


for it in feature_names:
    h.append(it)


# In[38]:


df = pd.read_csv('FEATURES1.txt', sep="\t", names=h)


# In[39]:


df.sort_values(by=['id'],inplace=True)


# In[40]:


X = np.zeros((1038,86))


# In[41]:


#read from df and store in feature matrix

for i in range(df.shape[0]):
    #pick the row
    row = df.loc[i].values.flatten().tolist()
    for j in range(0,86):
        X[i][j] = int(row[j+1])


# In[42]:


import scipy.sparse


# In[43]:


X_sp = scipy.sparse.csc_matrix(X)


# In[44]:


X_sp


# In[45]:


scipy.sparse.save_npz(str(path)+'X_sp.npz', X_sp)


# In[46]:


#create adjacency matrix

A = np.zeros((1038,1038))


# In[47]:


for i in range(T.shape[0]):
    v1 = int(T.iloc[i][0])
    v2 = int(T.iloc[i][2])
    
    A[v1][v2] = 1
    A[v2][v1] = 1


# In[48]:


A_sp = scipy.sparse.csc_matrix(A)


# In[49]:


scipy.sparse.save_npz(str(path)+'A_sp.npz', A_sp)


# In[50]:


#Create nodes keep and nodes hide


# In[51]:


nodes = [i for i in range(1038)]


# In[52]:


import random


# In[53]:


num_nodes_to_hide = int(0.1 * len(nodes))


# In[54]:


nodes_hide = random.sample(nodes , num_nodes_to_hide)


# In[55]:


#nodes that are not in nodes_hide will go in nodes_keep


# In[56]:


nodes_keep = []
for i in nodes:
    if i not in nodes_hide:
        nodes_keep.append(i)


# In[57]:


with open(str(path)+"nodes_keep.npy","wb") as nk:
    np.save(nk , nodes_keep)


# In[58]:


with open(str(path)+"nodes_hide.npy","wb") as nh:
    np.save(nh , nodes_hide)


# ## Use nodes_keep to create A_train and X_train

# In[59]:


#load A and X files


# In[60]:


import scipy.sparse as sp


# In[61]:


A = sp.load_npz(str(path)+"A_sp.npz")


# In[62]:


X = sp.load_npz(str(path)+"X_sp.npz")


# In[63]:


A = A.toarray()
X = X.toarray()


# In[64]:


tmp_X = X


# In[65]:


tmp_X = np.delete(tmp_X , tuple(nodes_hide) , axis = 0)


# In[66]:


X_train = sp.csc_matrix(tmp_X)


# In[67]:


sp.save_npz(str(path)+'ind_train_X.npz', X_train)


# In[ ]:





# In[68]:


tmp_A = A


# In[69]:


tmp_A = np.delete(tmp_A , tuple(nodes_hide) , axis = 0)
tmp_A = np.delete(tmp_A , tuple(nodes_hide) , axis = 1)


# In[70]:


A_train = sp.csc_matrix(tmp_A)
sp.save_npz(str(path)+'ind_train_A.npz' , A_train)


# In[ ]:





# In[ ]:





# In[71]:


## Create val_ones, val_zeros, test_ones, test_zeros, train_ones


# In[72]:


#train_ones are edges in A_train


# In[73]:


#load A_train

a_train = sp.load_npz(str(path)+'ind_train_A.npz')


# In[74]:


ta = a_train.toarray()


# In[75]:


train_ones = []
for i in range(ta.shape[0]):
    for j in range(ta.shape[1]):
        if ta[i][j] == 1:
            train_ones.append([i,j])


# In[76]:


train_ones = np.array(train_ones)


# In[77]:


#Use A file to create test_ones and test_zeros


# In[78]:


def get_train_data(A_train, batch_size ,inductive):
    nodes = []
    labels = []
    tmp_A = A_train.tolil()
    
    nodeNum = A_train.shape[0]
    
    
    while True:
        a = random.randint(0,nodeNum-1)
        b = random.randint(0,nodeNum-1)
        
        if not tmp_A[a,b]:
            nodes.append([a,b])

        if len(nodes) == batch_size:
            return nodes


# In[79]:


A = sp.load_npz(str(path)+'A_sp.npz')


# In[80]:


edges_in_A = []
A_arr = A.toarray()
for i in range(A_arr.shape[0]):
    for j in range(A_arr.shape[1]):
        if A_arr[i][j] == 1:
            edges_in_A.append([i,j])


# In[81]:


test_ones = random.sample(edges_in_A , int(0.1 * len(edges_in_A)))


# In[82]:


test_zeros = get_train_data(A , int(0.1 * len(edges_in_A)) , True)


# In[83]:


test_zeros = np.array(test_zeros)
test_ones = np.array(test_ones)


# In[84]:


#Create val_ones and val_zeros using A_train


# In[85]:


#load A_train


# In[86]:


A_train = sp.load_npz(str(path)+'ind_train_A.npz')


# In[87]:


edges_in_A_train = []
A_train_arr = A_train.toarray()
for i in range(A_train_arr.shape[0]):
    for j in range(A_train_arr.shape[1]):
        if A_train_arr[i][j] == 1:
            edges_in_A_train.append([i,j])


# In[88]:


batch_size_for_validation = int(0.1 * len(edges_in_A_train))


# In[89]:


val_ones = random.sample(edges_in_A_train , batch_size_for_validation)


# In[90]:


val_zeros = get_train_data(A_train , batch_size_for_validation , True)


# In[91]:


val_ones = np.array(val_ones)
val_zeros = np.array(val_zeros)


# In[92]:


outfile = open(str(path)+"pv0.10_pt0.00_pn0.10_arrays.npz","wb")


# In[93]:


np.savez(outfile,train_ones,val_ones, val_zeros, test_ones, test_zeros)


# In[ ]:





# In[94]:


#Create distance file using A_train


# In[95]:


dist_M = np.identity(A_train.shape[0])


# In[96]:


nodes_in = []
for i in range(len(edges_in_A_train)):
    nodes_in.append(edges_in_A_train[i][0])
    nodes_in.append(edges_in_A_train[i][1])
    
nodes_in = list(set(nodes_in))


# In[97]:


import networkx as nx


# In[98]:


tg = nx.Graph()
tg.add_edges_from(edges_in_A_train)


# In[99]:


for node in range(dist_M.shape[0]):
    if node in nodes_in:
        L = nx.single_source_shortest_path_length(tg,node)
        
        if len(L) > 0:
            for key in L.keys():
                if node != key:
                    dist_M[node][key] = L[key]


# In[100]:


for i in range(dist_M.shape[0]):
    for j in range(dist_M.shape[1]):
        if i!=j and dist_M[i][j]!=0:
            dist_M[i][j] = float("{0:.4f}".format((1/(dist_M[i][j]+1))))


# In[101]:


import torch


# In[102]:


dist_M = torch.from_numpy(dist_M)


# In[103]:


import pickle


# In[104]:


with open(str(path)+"dists-1.dat","wb") as dfile:
    pickle.dump(dist_M,dfile )

