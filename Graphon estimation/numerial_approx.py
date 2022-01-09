import numpy as np
from sympy import * 
import cvxpy as cp
import pandas as pd



def interval_X(X):
    '''
    X: sorted np.array of points sampled from Unif[0,1] 
    '''
    intervals=[]
    int_length=[]
    for i in range(len(X)-1):
        intervals.append((X[i],X[i+1]))
        int_length.append(X[i+1]-X[i])
        
    return int_length,intervals



def coefA(intl):
    '''
    intl: list of length of each interval
    return: the 'coefficient matrix in the system of equations' in 2d array
    '''
    n=len(intl)
    nrow=int((n**2+n)/2)
    ncol=int(n**2)
    #A=zeros(nrow,ncol)
    A=np.zeros(shape=(nrow,ncol))
    for i in range(n):
        A[i,]=np.array([0]*n*i+intl+[0]*n*(n-i-1))
        
    i=n
    for j in range(n-1):
        for k in range(j+1,n):
            A[i,j*n+k] = 1
            A[i,k*n+j] = -1
            i+=1
    return A


def alpha_estimate(Coef,centrality):
    '''
    Output: estimated alpha_ij
    Coef: the coefficient matrix
    centrality: array of centralities computed from the sampled graph
    '''
    n=len(centrality)-1
    c=centrality[:-1]
    y=np.array(list(c)+[0]*int((n**2-n)/2))
    
    result=np.linalg.lstsq(Coef, y)
    alpha1=result[0]
    
    matrix_C=Matrix(Coef)
    C_ns=matrix_C.nullspace()
    B=[np.reshape(np.array(a),(1,-1)) for a in C_ns]
    df=pd.DataFrame(B[0])
    for i in range(1,len(B)):
        dfi=pd.DataFrame(B[i])
        df =pd.concat([df,dfi],ignore_index=True)
    null_vecs=np.array(df)
    
    M=len(null_vecs)
    Var=cp.Variable(M)
    alphas=Var@null_vecs+alpha1
    cons=[a>=0 for a in alphas]+[a<=1 for a in alphas]
    
    obj = cp.Maximize(1) 
    #Create a problem with the objective and constraints
    prob = cp.Problem(obj, cons)

    #Solve problem, get optimal value
    val = prob.solve()
    if val==-np.inf:
        return "NO SOLUTION FOUND"
        
    var_hat=np.array(Var.value)
    
    alpha_hat=np.dot(var_hat,null_vecs)+alpha1
    
    return np.array(alpha_hat,dtype='float64')
   
    