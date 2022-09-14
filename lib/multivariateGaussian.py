import numpy as np
import scipy.special
import lib.generics

def logpdf_GAU_ND(x, mi, C):

    return -(x.shape[0]/2)*np.log(2*np.pi)-(0.5)*(np.linalg.slogdet(C)[1])- (0.5)*np.multiply((np.dot((x-mi).T, np.linalg.inv(C))).T,(x-mi)).sum(axis=0)

def joint_log_density(LS, priors):
    return LS + priors

def marginal_log_densities(jointProb):
    return lib.generics.mrow(scipy.special.logsumexp(jointProb, axis=0))

def log_posteriors (jointProb, marginals):
    return jointProb-marginals