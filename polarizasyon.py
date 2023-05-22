# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:14:25 2023

@author: omer_
"""
import math
import numpy as np
##Doğrusal Elektromanyetik Dalgaya Ait Fiziksel Nicelik Hesaplamaları
c=3e8
lambda_1=543e-9
f=c/lambda_1
t=1/f
k=2*math.pi*f
e0=100
pol_acisi=math.pi/4
dieleksbt=8.85e-12
I0=0.5*c*dieleksbt*e0*e0
def degree_pol(theta,given_pol):
    rotation_matrix=[[math.cos(theta),-math.sin(theta)],
                     [math.sin(theta),math.cos(theta)]]
    rotation_matrix_1=[[math.cos(theta),math.sin(theta)],
                     [-math.sin(theta),math.cos(theta)]]
    # matrix=[[math.pow(math.cos(theta), 2),math.sin(theta)*math.cos(theta)],
    #         [math.sin(theta)*math.cos(theta),math.pow(math.sin(theta), 2)]]
    matrix_1=np.matmul(given_pol,rotation_matrix_1)
    matrix=np.matmul(rotation_matrix,matrix_1)
    return matrix
pol=[[1,0],
     [0,0]]
radians=math.radians(90)
A=degree_pol(radians,pol)
horizontal_pol=10*np.asarray([[math.cos(radians)],[math.sin(radians)]])
E_out=np.matmul(A,horizontal_pol)
# image=[[degree_pol(math.radians(135),pol),degree_pol(math.radians(135),pol),degree_pol(math.radians(90),pol),degree_pol(math.radians(135),pol),degree_pol(math.radians(135),pol)],
#        [degree_pol(math.radians(135),pol),degree_pol(math.radians(135),pol),degree_pol(math.radians(90),pol),degree_pol(math.radians(135),pol),degree_pol(math.radians(135),pol)],
#        [degree_pol(math.radians(135),pol),degree_pol(math.radians(135),pol),degree_pol(math.radians(90),pol),degree_pol(math.radians(135),pol),degree_pol(math.radians(135),pol)],
#        [degree_pol(math.radians(135),pol),degree_pol(math.radians(135),pol),degree_pol(math.radians(90),pol),degree_pol(math.radians(135),pol),degree_pol(math.radians(135),pol)],
#        [degree_pol(math.radians(135),pol),degree_pol(math.radians(90),pol),degree_pol(math.radians(90),pol),degree_pol(math.radians(90),pol),degree_pol(math.radians(135),pol)]]
image=[[degree_pol(math.radians(45),pol),degree_pol(math.radians(135),pol)],
       [degree_pol(math.radians(90),pol),degree_pol(math.radians(0),pol)],
       ]
zeros=np.zeros((2,2))
lite_1=list()
for i in range(2):
    for j in range(2):
        final=np.matmul(image[i][j],horizontal_pol)
        lite_1.append(final)
        # final=np.matmul(final.T,final)
        # zeros[i,j]=final
e_out=image*horizontal_pol
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.optim as optim
labels = torch.tensor(0.2, dtype=torch.float32)
losses = []
epochs = 100
eta = 0.001
methods = ['backward', 'grad', 'optimizer']
method = methods[1]
weights = torch.zeros((2,2,2,2), dtype=torch.float32, requires_grad=True)
weights=torch.nn.init.xavier_uniform(weights)
if method == 'optimizer':
    optimizer = optim.SGD((weights),lr=eta)
m = torch.nn.Sigmoid()
total_1_loss=[]
for epoch in range(30):
    total_loss = 0
    qx=0
    for idx in range(1):
        # take current input
        X = torch.tensor(lite_1,dtype=torch.float32)
        y = labels
        
        for q1 in range(2):
            for q2 in range(2):
                
        # compute output and loss
                out =(torch.matmul(weights[q1,q2,:,:], X[qx]))
                out=torch.matmul(out.T,out)
                qx+=1
                loss = y-out
        # total_loss += loss.item()
                total_1_loss.append(loss.detach().numpy())
        
        
        if method == 'grad':
            gradw = torch.autograd.grad(loss, weights, retain_graph=True)
            with torch.no_grad():
                weights -= eta * gradw[0]
                
        
        
        elif method == 'backward':      
            # backpropagation
            loss.backward()
         
            # compute accuracy and update parameters
            with torch.no_grad():
                weights -= eta * weights.grad
                
                # reset gradient to zero
                weights.grad.zero_()
               
                
                
        elif method == 'optimizer':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    losses.append(total_loss / 1)
    #print(losses[-1])

plt.imshow(weights.detach().cpu().numpy())
# plot points, hyperplane and learning curve
plt.figure()
# plt.scatter(data[:,0].numpy(), data[:,1].numpy(), c=labels.numpy())
xr = np.linspace(0, 20, 10)
yr = (-1 / weights[1].item()) * (weights[0].item() * xr  )
plt.plot(xr, yr,'-')
plt.show()

plt.figure()
plt.plot(losses, '-')
plt.show()