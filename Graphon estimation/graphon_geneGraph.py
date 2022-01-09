import numpy as np
from math import log, exp, sin, cos,pi



#-------------------------------------------------------#
# graphon functions
#-------------------------------------------------------#

def w1(u,v):
    K=int(log(1000))
    if(u==1):
        k_u=K
    else:
        k_u=int(u*K)+1
    if(v==1):
        k_v=K
    else:
        k_v=int(v*K)+1
    if(k_u==k_v):
        k=k_u
        return k/(K+1)
    else:
        return 0.3/(K+1)
    
def w2(u,v):
    return sin(5*pi*(u+v-1)+1)/2+0.5

def w3(u,v):
    return 1-1/(1+exp(15*(0.8*abs(u-v))**(4/5)-0.1))

def w4(u,v):
    return (u**2+v**2)/3*cos(1/(u**2+v**2))+0.15 



# Collection of graphon functions
def W1(x,y):
    return x*y

def W2(x,y):
    w=math.exp(-max(x,y)**0.75)
    return w

def W3(x,y):
    w=math.exp(-0.5*(min(x,y)+x**0.5+y**0.5))
    return w

def W4(x,y):
    return abs(x-y)

def W_fr(x,y):
    return (x**2+y**2)/2


def W_gs(x,y):
    w=min(x,y)*(1-max(x,y))
    return w



######### Generate sampled graphon #######

def prob_matrix(U,W_func):
    """
    U: is the latent variables
    """
    N=len(U)
    
    probMat=np.zeros(shape=(N,N),dtype='float64')
    
    for i in range(N):
        for j in range(i, N):
            x=U[i]
            y=U[j]
            probMat[i,j]=W_func(x,y)
            probMat[j,i]=probMat[i,j]
            
    return probMat


def sampled_graph(U,W_func):
    '''
    U: is the latent variables
    W_func: the graphon function
    '''
    N=len(U)
    
    Adj=np.zeros(shape=(N,N),dtype='float64')
    for i in range(N):
        for j in range(i,N):
            x=U[i]
            y=U[j]
            w_xy=W_func(x,y)
            Adj[i,j]=np.random.binomial(n=1,p=w_xy)
            Adj[j,i]=Adj[i,j]
    return Adj  
