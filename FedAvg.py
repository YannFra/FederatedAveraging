#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt

import syft as sy
hook=sy.TorchHook(torch)


def FedAvg(model,server,clients,X,y,iter_max,epochs,epsilon,
    loss_f=nn.MSELoss()):
    
    """
    NN model structure used by the clients and the server
    server virtualworker created for the server
    clients list of the K virtual workers Ck
    X list of the K clients feature dataset Xk
    y list of the K clients feature dataset yk
    iter_max: integer. Maximal server iterations number
    epochs: integer. Number of epochs run on the client side.
    epsilon: float. Does a step if its loss improvment is superior to epsilon
    loss_f: loss function used
    
    
    returns :
        - Global model of the server : Pytorch NN
        - Loss history of the clients and the server : list
        - Model Parameters evolution : list
    """
        
    #Variables initialization
    K=len(clients) #number of clients
    
    
    #Checking the quality of the input
    if len(X)!=K or len(y)!=K:
        print("The dimension of clients, X, and y is not the same")
    
    
    #Metrics we are interested in 
    loss_hist=[]
    a_hist=[]
    b_hist=[]
    
    
    #Send each dataset to the associated client    
    X_sent=[]
    y_sent=[]
    for k in range(K):
        X_sent.append(X[k].send(clients[k]))
        y_sent.append(y[k].send(clients[k]))
    
    
    #Variables initialization
    precision=100 #Loss variation from one step to another
    i=0 #iteration number
    
    while precision>epsilon and i<iter_max:
        
        #List containing the model sent to each client
        models=[]
        #List containing the optimizers of each client
        optimizers=[]
        
        for client in clients:
            models+=[model.copy().send(client)]
            optimizers+=[optim.SGD(params=models[-1].parameters(),lr=0.1)]
        
        
        for j in range(epochs):
            
            #Losses of each client when they have finished updating a local model
            clients_losses=[]
            
            
            #Update each client
            for k in range(K):
            
                optimizers[k].zero_grad()
                y_pred = models[k](X_sent[k])
                client_loss = loss_f(y_pred,y_sent[k])
                client_loss.backward()
        
                optimizers[k].step()
                
                clients_losses.append(client_loss.get().data.numpy())
                    

        loss_hist.append(clients_losses)
        
        
        try:precision=(sum(loss_hist[-1])-sum(loss_hist[-2]))**2
        except:pass
        
    
        #Upload on the server the client's model
        for model_k in models:
            model_k.move(server)
    
        
        with torch.no_grad():
            
            #Save each client's model parameters
            clients_a=[]
            clients_b=[]
            new_weight=0*model.weight.data
            new_bias=0*model.bias.data
            for model_k in models: 
                mod_weight=model_k.weight.get().data
                mod_bias=model_k.bias.get().data
                
                
                clients_a.append(np.reshape(mod_weight.numpy(),1))
                clients_b.append(mod_bias.numpy())
                
                new_weight+=mod_weight
                new_bias+=mod_bias

            
            a_hist.append(clients_a)
            b_hist.append(clients_b)
            
            
            #The server creates the new global model
            new_weight/=K
            new_bias/=K
            
            model.weight.set_(new_weight)
            model.bias.set_(new_bias)
        
        if i%10==0:
            print("iteration "+str(i)+", Model Loss:"+ str(sum(loss_hist[-1])))
        
        i+=1


    return model,np.array(loss_hist),np.array(a_hist),np.array(b_hist)





def get_loss(loss_hist):


    plt.figure(figsize=(15,6))
    
    for k in range(len(loss_hist[0])):
        
        Ck_loss=[loss_hist[i][k]for i in range(len(loss_hist))]
    
        plt.plot(Ck_loss,label="C"+str(k+1)+" Loss")
        
    Server_loss=[np.mean(step_loss) for step_loss in loss_hist ]   
    plt.plot(Server_loss,label="Server Loss")
    
    plt.title("Loss of the different clients and server at a given iteration")
    plt.xlabel("Server iteration")
    plt.ylabel("Loss")
    plt.legend()
    
    

def weights_plot(a_hist,b_hist):
    plt.figure(figsize=(20,10))
    
    plt.subplot(1,2,1)
    
    for k in range(len(a_hist[0])):
        
        Ck_a=[a_hist[i][k]for i in range(len(a_hist))]
        
        plt.plot(Ck_a,label="C"+str(k+1))
    
    S_a=[np.mean(a) for a in a_hist]
    plt.plot(S_a,label="Server")
        
    plt.title("Clients' weight evolution")
    plt.xlabel("Iterations")
    plt.ylabel("Weight value")
    plt.legend()
    
    
    plt.subplot(1,2,2)
    for k in range(len(a_hist[0])):
        
        Ck_b=[b_hist[i][k]for i in range(len(b_hist))]
        
        plt.plot(Ck_b,label="C"+str(k+1))

    S_b=[np.mean(b) for b in b_hist]
    plt.plot(S_b,label="Server")
    
    plt.title("Clients' bias evolution")
    plt.xlabel("Iterations")
    plt.ylabel("Bias value")
    plt.legend()
    
    
    
def gradient_ratio_plot(a_hist,b_hist):

    plt.figure(figsize=(20,10))
    
    plt.subplot(1,2,1)
    a1_hist=[a_hist[i][0] for i in range(len(a_hist))]
    a2_hist=[a_hist[i][1] for i in range(len(a_hist))]    
    plt.scatter(a1_hist,a2_hist,c=[[i] for i in range(len(a_hist))],cmap='plasma')
    plt.colorbar()
    
    plt.title("Weigth relationship for each iteration")
    plt.xlabel("Gradient Weight Client 1")
    plt.xlabel("Gradient Weight Client 2")
    
    plt.subplot(1,2,2)
    b1_hist=[b_hist[i][0] for i in range(len(a_hist))]
    b2_hist=[b_hist[i][1] for i in range(len(a_hist))]  
    plt.scatter(b1_hist,b2_hist,c=[[i] for i in range(len(b_hist))],cmap='plasma')
    plt.colorbar()
    plt.title("Bias relationship for each iteration")
    plt.xlabel("Gradient Bias Client 1")
    plt.xlabel("Gradient Bias Client 2")

