import numpy as np
from math import cos, pi,sin,exp,log, sqrt
from numpy import random 
from scipy.stats import norm
from statistics import stdev

#-------------------------------------------------------#
#  feature maps
#-------------------------------------------------------#
# feature 1
def f1(x):
    return 2*cos(2*pi*((1-x)**2)) # not monotone

# feature 2
def f2(x):
    return 10*(x**2)-12*x+5   # not monotone

# feature 3
def f3(x):
    return 2*cos(pi*x)  # monotone

# feature 4
def f4(x):
    return 2/3*norm.ppf(x)  # monotone


#-------------------------------------------------------#
# get the feature for all nodes
#-------------------------------------------------------#

def featureMatrix(U,fmaps,sigma):
    '''
    U: points sampled from uniform(0,1)
    fmaps: list of feature maps
    sigma: sd for randomness
    '''
    n=len(U)
    m=len(fmaps)
    X=np.zeros(shape=(n,m),dtype='float64')
    for i in range(m):
        fi=fmaps[i]
        fu=np.array([fi(u) for u in U])
        std_fu=stdev(fu)
        X[:,i]=fu/std_fu+random.normal(0,sigma,n)
    return X



