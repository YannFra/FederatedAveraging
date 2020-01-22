#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In[2]
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt

import syft as sy
hook=sy.TorchHook(torch)

from FedAvg import *

# In[2]:


#Generate two simple linear model with similar coefficients
#torch.rand => uniform distribution

dim=100

X1=torch.rand(dim, 1) 
X2=torch.rand(dim,1)

a_1,b_1=5,-2
a_2,b_2=4,-1
y1=a_1*X1+b_1+0.5*torch.randn(dim,1)
y2=a_2*X2+b_2+0.5*torch.randn(dim,1)


# In[2]:
#Create the server and the clients
server = sy.VirtualWorker(hook, id="server")
C1 = sy.VirtualWorker(hook, id="C1")
C2 = sy.VirtualWorker(hook, id="C2")

#Send the features and the labels to its associated client
C1_x = X1.send(C1)
C1_y = y1.send(C1)

C2_x = X2.send(C2)
C2_y = y2.send(C2)

# In[2]:
model = nn.Linear(1,1)

clients=[C1,C2]
features=[X1,X2]
labels=[y1,y2]


iter_max=100
epochs=2
epsilon=10**-10



loss_hist,a_hist,b_hist=FedAvg(model,server,clients,features,labels,iter_max,epochs,epsilon)


# In[2]:
get_loss(loss_hist)
# In[2]:
weights_plot(a_hist,b_hist)
# In[2]:
gradient_ratio_plot(a_hist,b_hist)


# In[2]:





