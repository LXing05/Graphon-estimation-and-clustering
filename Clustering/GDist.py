import numpy as np
import math
from numpy.random import binomial
from scipy.integrate import quad
import torch

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
            Adj[i,j]=binomial(n=1,p=w_xy)
            Adj[j,i]=Adj[i,j]
    return Adj  

#----------------------------------------------------------------------#
#                   Distance between graphs                            #
#----------------------------------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hist_apprx(graphs, n0=30):
    graphs_appr = []
    for graph in graphs:
        # degree sort
        nn = graph.shape[0]
        h = int(nn / n0)

        deg = torch.sum(graph, axis=1)
        id_sort = torch.argsort(-deg)

        graph_sorted = graph[id_sort]
        graph_sorted = graph_sorted[:, id_sort]

        # histogram approximation
        graph_apprx = torch.zeros((n0, n0), dtype=torch.float64).to(device=device)
        for i in range(n0):
            for j in range(i + 1):
                graph_apprx[i][j] = torch.sum(graph_sorted[i * h:i * h + h, j * h:j * h + h]) / (h * h)
                graph_apprx[j][i] = graph_apprx[i][j]

        graphs_appr.append(graph_apprx)

    return graphs_appr


def distance_matrix(all_graphs,n0):
    m = len(all_graphs)
    dist = torch.zeros((m, m), dtype=torch.float64).to(device=device)
    for i in range(m):
        for j in range(i + 1):
            dist[i][j] = (torch.norm(all_graphs[i] - all_graphs[j]))/n0 # /all_graphs[i].shape[0] not needed
            dist[j][i] = dist[i][j]
    return dist

#--------------------------------------------------------#
#             Distance between graphons                  #
#--------------------------------------------------------#
def W_dif(x,y,w1,w2):
    w1_xy=w1(x,y)
    w2_xy=w2(x,y)
    return (w1_xy-w2_xy)**2

def IntgX(y,w1,w2):
    intg=quad(W_dif,0,1,args=(y, w1,w2))[0]
    return intg



def Graphon_distmat(graphons):
    '''
    input: list of graphon functions
    '''
    n=len(graphons)
    distmat=np.zeros(shape=(n,n),dtype='float64')   
    for i in range(n-1):
        w_i=graphons[i]
        for j in range(i+1,n):
            w_j=graphons[j]
            distmat[i,j]=np.sqrt(quad(lambda y: IntgX(y, w_i,w_j), 0, 1)[0])
            distmat[j,i]=distmat[i,j]
    return distmat



