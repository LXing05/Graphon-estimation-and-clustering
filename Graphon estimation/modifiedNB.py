import numpy as np
from statistics import stdev
from numpy import transpose as tr
from math import sqrt,log


def get_d0(A):
    '''
    A: 2d array, adjacency matrix of graph
    '''
    n=len(A)
    d0=np.zeros(shape=(n,n))
    A_sq=A @ A
    for i in range(n-1):
        for j in range(i+1,n):
            dif_ij=abs(A_sq[i,:]-A_sq[j,:])
            dif=np.delete(dif_ij,[i,j])
            d0[i,j]=max(dif+np.random.uniform(0,1,1)/n)/n
            d0[j,i]=d0[i,j]
    return d0



def get_s(X):
    '''
    X: 2d array, matrix of all features for each node. nrow(X)=number of nodes
    '''
    n,m=np.shape(X)
    s=np.zeros(shape=(n,n))
    X_sq=X @ tr(X)
    for i in range(n-1):
        for j in range(i+1,n):
            dif_ij=abs(X_sq[i,:]-X_sq[j,:])
            dif=np.delete(dif_ij,[i,j])
            s[i,j]=max(dif+np.random.uniform(0,1,1)/n)/m
            s[j,i]=s[i,j]
    return s


def neighborMat(A,X,l):
    '''
    A: adjacency matrix
    X: feature matrix
    l: constant, for computing d
    Output: the neighbor of each node 
    which(NB[i,]!=0) # Neighbor of node i
    '''
    n=len(A)
    d0=get_d0(A)
    s=get_s(X)
    d = d0/d0.max() + l*s/s.max()
    NB=np.zeros(shape=(n,n))
    h = sqrt(log(n)/n)
    for i in range(n):
        di=d[i,:]
        q_ih=np.percentile(np.delete(di,i),h*100)
        NB[i,]=np.array([int(d<=q_ih) for d in di])
        if sum(NB[i,:])>1: 
            NB[i,i]=0
    return NB


def new_deg(A,X,l):
    '''
    degree based on the modified neighborhood
    '''
    NB=neighborMat(A,X,l)
    return np.diag(NB@A)/len(A)





