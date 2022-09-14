import lib.generics 
import numpy as np

def computePCA(C, D, m):
    # Get the eigenvalues (s) and eigenvectors (columns of U) of C
    s, U = np.linalg.eigh(C)
    # Principal components
    P = U[:, ::-1][:, 0:m]
    # PCA projection matrix
    DP = np.dot(P.T, D)
    return DP, P

def PCA(D, m):
    # L is only needed to plot if m=2, PCA is unsupervised
    DC = lib.generics.centeredData(D)
    C = (1/DC.shape[1]) * (np.dot(DC, DC.T)) #Covariance matrix
    return computePCA(C, D, m)