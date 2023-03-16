import numpy
import scipy.special
import lib.generics as generics
from lib.generics import mrow, mcol

def logpdf_GMM(D, gmm):
    S = numpy.zeros((len(gmm), D.shape[1]))
    for g in range(len(gmm)):
        (w, mu, C) = gmm[g]
        S[g, :] = logpdf_GAU_ND(D, mu, C) + numpy.log(w)
    logD = scipy.special.logsumexp(S, axis=0)
    return S, logD


def logpdf_GAU_ND(D, mu, sigma):
    P = numpy.linalg.inv(sigma)
    c1 = 0.5 * D.shape[0] * numpy.log(2 * numpy.pi)
    c2 = 0.5 * numpy.linalg.slogdet(P)[1]
    c3 = 0.5 * (numpy.dot(P, (D - mu)) * (D - mu)).sum(0)
    return - c1 + c2 - c3


def empirical_mean(X):
    return mcol(X.mean(1))

def empirical_covariance(X):
    mu = empirical_mean(X)
    C = numpy.dot((X - mu), (X - mu).T) / X.shape[1]
    return C

def GMM_LBG(D, alpha, components, psi, type):
    gmm = [(1, empirical_mean(D), empirical_covariance(D))]
    while len(gmm) <= components:
        gmm = GMM_EM(D, gmm, psi, type)
        if(len(gmm) == components):
            break
        newGmm = []
        for i in range(len(gmm)):
            (w, mu, sigma) = gmm[i]
            U, s, Vh = numpy.linalg.svd(sigma)
            d = U[:, 0:1] * (s[0] ** 0.5) * alpha
            newGmm.append((w / 2, mu + d, sigma))
            newGmm.append((w / 2, mu - d, sigma))
        gmm = newGmm
    return gmm


def GMM_EM(DT, gmm, psi, mode, diff = 1e-6):
    
    D, N = DT.shape
    to = None
    tn = None

    while to == None or tn - to > diff:
        to = tn
        S, logD = logpdf_GMM(DT, gmm)
        tn = logD.sum() / N
        P = numpy.exp(S - logD)

        newGmm = []
        sigmaTied = numpy.zeros((D, D))
        for i in range(len(gmm)):
            gamma = P[i, :]
            Z = gamma.sum()
            F = (mrow(gamma) * DT).sum(1)
            S = numpy.dot(DT, (mrow(gamma) * DT).T)
            w = Z/P.sum()
            mu = mcol(F / Z)
            sigma = (S / Z) - numpy.dot(mu, mu.T)
            if mode == 'GMMTied':
                sigmaTied += Z * sigma
                newGmm.append((w, mu))
                continue
            elif mode == 'GMMDiag':
                sigma *= numpy.eye(sigma.shape[0])
            U, s, _ = numpy.linalg.svd(sigma)
            s[s<psi] = psi
            sigma = numpy.dot(U, mcol(s) * U.T)
            newGmm.append((w, mu, sigma))

        if mode == 'GMMTied':
            sigmaTied /= N
            U, s, _ = numpy.linalg.svd(sigmaTied)
            s[s<psi] = psi
            sigmaTied = numpy.dot(U, mcol(s) * U.T)
            newGmm2 = []
            for i in range(len(newGmm)):
                (w, mu) = newGmm[i]
                newGmm2.append((w, mu, sigmaTied))
            newGmm = newGmm2
        gmm = newGmm

    return gmm


class GMM:
    
    def __init__(self, components, mode):
        self.components = components
        self.mode = mode

    def train(self, D, L, psi=1e-2, alpha=1e-1):
        D0 = D[:, L == 0]
        D1 = D[:, L == 1]

        self.gmm0 = GMM_LBG(D0, alpha, self.components, psi, self.mode)
        self.gmm1 = GMM_LBG(D1, alpha, self.components, psi, self.mode)

        return self


    def predictAndGetScores(self, D):
        S, logD0 = logpdf_GMM(D, self.gmm0)
        S, logD1 = logpdf_GMM(D, self.gmm1)
        return logD1 - logD0