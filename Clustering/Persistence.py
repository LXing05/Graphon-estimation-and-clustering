import numpy as np
import pandas as pd
from Bio import SeqIO
import pickle
from ripser import ripser
#import multiprocessing
#from joblib import Parallel, delayed
#from tqdm import tqdm
import sys
from gudhi.wasserstein import wasserstein_distance
import gudhi.hera.bottleneck as bottleneck
import ot
#from numba import cuda

def perDiag(x, order):
    '''
    gennerate persistent diagrams 
    input: 
    x: the data
    order:  0 means H0, 1 means H1
    '''
    diagrams = ripser(x)['dgms']
    return diagrams[order][~np.isinf(diagrams[order]).any(axis=1)]


def barcode(x, order):
    '''
    input: 
    x: list of data representations 
    order: 0 means H0, 1 means H1
    return: list of barcoode
    '''
    num_cores = multiprocessing.cpu_count()
    inputs = tqdm(x)
    processed_list = Parallel(n_jobs=num_cores)(delayed(perDiag)(i, order) for i in inputs)
    return processed_list




def wasserstein_dist_matrix(Diags, orderp = 1, interp = np.inf): 
    '''Function to compute the wasserstein distance matrix from Diagrams Barcode: numpy version'''
    n = len(Diags)
    Dist_Mat = np.zeros(shape=(n,n), dtype='float64')
    for i in range(n-1):
        for j in range(i+1,n):
            Dist_Mat[i,j] = wasserstein_distance(Diags[i], Diags[j], order = orderp, internal_p = interp)
    Dist_Mat = Dist_Mat + np.transpose(Dist_Mat)
    return Dist_Mat 



def bottleneck_dist_matrix(Diags, epsilon = 10**(-5)):
    '''Function to compute the bottleneck distance (using optimized algorithm) with controling on epsilon error'''
    n = len(Diags)
    Dist_Mat = np.zeros(shape = (n,n),dtype = 'float64')
    for i in range(n-1):
        for j in range(i+1, n):
            Dist_Mat[i,j] =bottleneck.bottleneck_distance(Diags[i], Diags[j], delta = epsilon)
    Dist_Mat = np.transpose(Dist_Mat) + Dist_Mat
    return Dist_Mat




def wasserstein_dist(graphPdiags, graphonPdiag,orderp = 1, interp = np.inf): 
    '''
    Function to compute the wasserstein distance between sampled graphs and graphs
    graphPdiags: the list of persistence diagrams of sampled graphs (with different number of nodes)
    graphonPdiag: the persistence diagram of graphons
    '''
    n = len(graphPdiags)
    Dist = np.zeros(n, dtype='float64')
    for i in range(n):
        Dist[i] = wasserstein_distance(graphPdiags[i],graphonPdiag, order = orderp, internal_p = interp)
    return Dist



def bottleneck_dist(graphPdiags, graphonPdiag, epsilon = 10**(-5)):
    '''
    Function to compute the wasserstein distance between sampled graphs and graphs
    graphPdiags: the list of persistence diagrams of sampled graphs (with different number of nodes)
    graphonPdiag: the persistence diagram of graphons
    '''
    n = len(graphPdiags)
    Dist = np.zeros(n, dtype='float64')
    for i in range(n):
        Dist[i] =bottleneck.bottleneck_distance(graphPdiags[i],graphonPdiag, delta = epsilon)
    return Dist



def wasDist_torch(Diags,orderp = 1, interp=np.inf,autodiff = True, device_type = 'gpu'):
    n = len(Diags)
    matrix  = torch.zeros(n, n, dtype = torch.float64,device = device_type)
    for i in range(n - 1):
        for j in range(i+1, n):
            matrix[i,j] = wasserstein_distance(Diags[i],Diags[j], order_p = orderp , 
                                                internal_p = interp , enable_autodiff=autodiff)
    matrix  = matrix + torch.transpose(matrix,0,1) 
    return matrix