#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np 
import json
import re
import os


# In[29]:


df = pd.read_csv("TRAIN_TEST_MERGE_UNIQUE.csv")


# In[30]:


counter = 0


# In[31]:


test_df = pd.DataFrame(columns=["SUBJECT","RELATION","OBJECT"])


# In[32]:


def func_helper(index,df,row,indices_to_del):
    s = row[0]
    o = row[2]
    nodes = []
    for i in range(df.shape[0]):
        if i != index and i not in indices_to_del:
            nodes.append(df.iloc[i,0])
            nodes.append(df.iloc[i,2])
    
    if s in nodes and o in nodes:
        return True
    else:
        return False


# In[33]:


num_edges_to_sample = int(0.10 * len(df))


# In[34]:


num_edges_to_sample


# In[ ]:





# In[35]:


df = df.sample(frac=1)
df = df.reset_index()
df.drop(['index'],axis=1,inplace=True)


# In[37]:


indices_to_del = []
for i in range(df.shape[0]):
    row = df.loc[i]
    if func_helper(i,df,row,indices_to_del):
        count = len(test_df)
        test_df.loc[count] = row
        counter = counter + 1
        if counter == num_edges_to_sample:
            break
        indices_to_del.append(i)
    


# In[24]:


test_df.to_csv("similarity/input/TEST_SET_SAMPLED.csv",index=False)


# In[26]:


X = df.drop(indices_to_del)


# In[27]:


X.to_csv("similarity/input/TRAIN_SET_SAMPLED.csv",index=False)


# In[ ]:




