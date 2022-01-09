import numpy as np
from math import log, exp, sin, cos,pi
from scipy.integrate import quad
from scipy.stats import norm
from numpy.random import normal, uniform, binomial

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


def w5(u,v):
    w=exp(-max(u,v)**0.75)
    return w

def w6(x,y):
    w=exp(-0.5*(min(x,y)+x**0.5+y**0.5))
    return w

def w7(x,y):
    return (x**2+y**2)/2 

def w8(x,y):
    w=min(x,y)*(1-max(x,y))
    return w

def w9(x,y):
    return abs(x-y)

def w10(x,y):
    return x*y
#-------------------------------------------------------#
#  feature maps
#-------------------------------------------------------#

def f1(x):  
    return 2*cos(2*pi*(1-x)**2)

def f2(x):
    return 10*x**2-12*x+5

def f3(x):
    return 2*cos(pi*x)

def f4(x):
    return 2/3*norm.ppf(x, loc=0, scale=1)



#-------------------------------------------------------#
# get the feature for all nodes
#-------------------------------------------------------#
def featureMatrix(U,fmaps,sigma):
    n=len(U)
    m=len(fmaps)
    X=np.zeros(shape=(n,m),dtype='float64')
    for i in range(m):
        f=fmaps[i]
        f_U=np.array(list(map(f,U)))
        sd=np.std(f_u)
        X[:,i]=f_U/sd+normal(loc=0.0, scale=sigma, size=1)
    return(X)
 


    












