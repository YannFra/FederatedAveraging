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
X3=torch.rand(dim,1)

a_1,b_1=10,5
a_2,b_2=6,-1
a_3,b_3=2,-0.8
a=[a_1,a_2,a_3]
b=[b_1,b_2,b_3]

y1=a_1*X1+b_1+0.5*torch.randn(dim,1)
y2=a_2*X2+b_2+0.5*torch.randn(dim,1)
y3=a_3*X3+b_3+0.5*torch.randn(dim,1)


# In[2]:
#Create the server and the clients
server = sy.VirtualWorker(hook, id="server")
C1 = sy.VirtualWorker(hook, id="C1")
C2 = sy.VirtualWorker(hook, id="C2")
C3 = sy.VirtualWorker(hook, id="C3")

# In[2]:
model = nn.Linear(1,1)

clients=[C1,C2,C3]
features=[X1,X2,X3]
labels=[y1,y2,y3]


iter_max=100
epochs=5
epsilon=10**-10



model,loss_hist,a_hist,b_hist=FedAvg(model,server,clients,features,labels,iter_max,epochs,epsilon)


# In[2]:
get_loss(loss_hist)
# In[2]:
weights_plot(a_hist,b_hist)
# In[2]:
#gradient_ratio_plot(a_hist,b_hist)


# In[2]:

def display_distribution(features,labels,a,b):
    
    x_axis=np.linspace(0,1,100)

    plt.figure(figsize=(20,10))
    
    
    for k in range(len(a)):
        
        Xk=features[k].detach().numpy()
        yk=labels[k].detach().numpy()
        
        plt.scatter(Xk,yk,label="Client "+str(k+1))
        plt.plot(x_axis,a[k]*x_axis+b[k],label="Ground Truth C"+str(k+1))
        
        
        plt.scatter(Xk,model(features[k]).detach().numpy(),label="Global Model Prediction Client "+str(k))
        
            
    plt.legend()
    plt.show()

display_distribution(features,labels,a,b)
    

# In[4]:


def get_gaussian(param_hist):
    """param_hist is a list of list.
    Each sublist is the parameter for each parameter at step t
    """
    
    mu=np.array([np.mean(sublist) for sublist in param_hist])
    
    sigma=np.array([np.mean((sublist-mu)**2) for sublist in param_hist])

    return mu,sigma



def plot_parameter_evolution(param_hist):
    """
    
    """
    mu,sigma=get_gaussian(param_hist)
    
    plt.figure(figsize=(20,10))
    
    plt.subplot(2,2,1)
    
    plt.plot(mu, label="Global Model Evolution")
    
    for k in range(len(param_hist[0])):        
        plt.plot(param_hist[:,k],label="C"+str(k+1))
    
    plt.title("Parameter Mean Evolution")
    plt.xlabel("Iterations")
    plt.ylabel("Weight value")
    plt.legend()
    
    plt.subplot(2,2,2)
    
    plt.plot(sigma, label="Global Model Evolution")
    
    plt.title("Parameter Sigma Evolution")
    plt.xlabel("Iterations")
    plt.ylabel("Variance Evolution")
    plt.legend()
    
    plt.show()
    
    plt.subplot(2,2,3)
    
    for k in range(len(param_hist[0])):        
        plt.plot(np.reshape(param_hist[:,k],-1)-mu,label="C"+str(k+1))
    
    plt.title("Parameter Bias Evolution")
    plt.xlabel("Iterations")
    plt.ylabel("Parameter Value - Parameter Mean")
    plt.legend()
    
    plt.show()
    
    plt.subplot(2,2,4)
    
    
    for i in range(len(param_hist[0])):
        Ci=param_hist[:,i]
        for j in range(i,len(param_hist[0])):
            if i!=j:
                Cj=param_hist[:,j]
                
                plt.scatter(Ci,Cj, s=10,label="x-axis: C"+str(i+1)+" y-axis: C"+str(j+1))
    
    
    plt.title("Parameter Time Evolution")
    plt.xlabel("Parameter Value")
    plt.ylabel("Parameter Value")
    plt.legend()
    
    plt.show()
    
plot_parameter_evolution(a_hist)
plot_parameter_evolution(b_hist)









    



