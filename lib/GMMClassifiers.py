import numpy
import lib.generics as generics
import scipy.special 
import lib.multivariateGaussian as multivariateGaussian

class GMM:
    
    
    def __init__(self, components):
        self.components = components
    
    def train (self, D, L):
        D0 = D[:, L==0]
        D1 = D[:, L==1]
       
        
        #represent GMM as a list of parameters: weight, mean, Covariance matrix  
        #We can use the Maximum Likelihood solution for a Gaussian density as starting point:    
        GMM0_init = [(1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), generics.constrainSigma(numpy.cov(D0).reshape((D0.shape[0], D0.shape[0]))))]
        GMM1_init = [(1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), generics.constrainSigma(numpy.cov(D1).reshape((D1.shape[0], D1.shape[0]))))]
       
        self.GMM0 = LBGalgorithm(GMM0_init, D0, self.components)
        self.GMM1 = LBGalgorithm(GMM1_init, D1, self.components)
        
        
    def predict (self, X):
        PD0 = compute_posterior_GMM(X, self.GMM0)
        PD1 = compute_posterior_GMM(X, self.GMM1)
 
        PD = numpy.vstack((PD0,PD1))
        return numpy.argmax(PD, axis=0)

    def predictAndGetScores(self, X):
        LS0 = computeLogLikelihood(X, self.GMM0)
        LS1 = computeLogLikelihood(X, self.GMM1)
        llr = LS1-LS0
        return llr
    
    def fastTraining(self, D, L, GMM0, GMM1):
        D0 = D[:, L==0]
        D1 = D[:, L==1]
        self.GMM0 = LBGalgorithm (GMM0, D0, 1)
        self.GMM1 = LBGalgorithm (GMM1, D1, 1)
        return [self.GMM0, self.GMM1]


class GMMDiag:
    
    
    def __init__(self, components):
        self.components = components
    
    def train (self, D, L):
        D0 = D[:, L==0]
        D1 = D[:, L==1]
       
       
        GMM0_init = [(1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), generics.constrainSigma(numpy.cov(D0)*numpy.eye( D0.shape[0]).reshape((D0.shape[0]),D0.shape[0])))]
        GMM1_init = [(1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), generics.constrainSigma(numpy.cov(D1)*numpy.eye( D1.shape[0]).reshape((D1.shape[0]),D1.shape[0])))]
       

        self.GMM0 = DiagLBGalgorithm (GMM0_init, D0, self.components)
        self.GMM1 = DiagLBGalgorithm (GMM1_init, D1, self.components)
     
        
        
    def predict (self, X):

        
        PD0 = compute_posterior_GMM(X, self.GMM0)
        PD1 = compute_posterior_GMM(X, self.GMM1)
      
 
        PD = numpy.vstack((PD0,PD1))
        return numpy.argmax(PD, axis=0)
    
    def predictAndGetScores(self, X):
    
        LS0 = computeLogLikelihood(X, self.GMM0)
        LS1 = computeLogLikelihood(X, self.GMM1)
        
        llr = LS1-LS0
        return llr

    def fastTraining(self, D, L, GMM0, GMM1):
        D0 = D[:, L==0]
        D1 = D[:, L==1]
        
        self.GMM0 = DiagLBGalgorithm (GMM0, D0, 1)
        self.GMM1 = DiagLBGalgorithm (GMM1, D1, 1)
        
        return [self.GMM0, self.GMM1]


class GMMTiedCov:
    
    def __init__(self, components):
        self.components = components
    
    
    def train (self, D, L):
        D0 = D[:, L==0]
        D1 = D[:, L==1]
       
        sigma0 =  numpy.cov(D0).reshape((D0.shape[0], D0.shape[0]))
        sigma1 =  numpy.cov(D1).reshape((D1.shape[0], D1.shape[0]))
        
        self.sigma = 1/(D.shape[1])*(D[:, L == 0].shape[1]*sigma0+D[:, L == 1].shape[1]*sigma1)
        GMM0_init = [(1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), generics.constrainSigma(self.sigma))]
        GMM1_init = [(1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), generics.constrainSigma(self.sigma))]
       

        self.GMM0 = TiedLBGalgorithm (GMM0_init, D0, self.components)
        self.GMM1 = TiedLBGalgorithm (GMM1_init, D1, self.components)
        
        
        
    def predict (self, X): 
      
        PD0 = compute_posterior_GMM(X, self.GMM0)
        PD1 = compute_posterior_GMM(X, self.GMM1)
     
 
        PD = numpy.vstack((PD0,PD1))
        return numpy.argmax(PD, axis=0)

    def predictAndGetScores(self, X):
    
        LS0 = computeLogLikelihood(X, self.GMM0)
        LS1 = computeLogLikelihood(X, self.GMM1)
        
        llr = LS1-LS0
        return llr
    
    def fastTraining(self, D, L, GMM0, GMM1):
        D0 = D[:, L==0]
        D1 = D[:, L==1]
        
        self.GMM0 = LBGalgorithm.TiedLBGalgorithm (GMM0, D0, 1)
        self.GMM1 = LBGalgorithm.TiedLBGalgorithm (GMM1, D1, 1)
        
        return [self.GMM0, self.GMM1]


def logpdf_GMM(X, gmm):
    S = numpy.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):
        # Compute log density
        S[i, :] = multivariateGaussian.logpdf_GAU_ND(X, gmm[i][1], gmm[i][2])
    return S

def joint_log_density_GMM (S, gmm):
    
    for i in range(len(gmm)):
        # Add log of the prior of the corresponding component
        S[i, :] += numpy.log(gmm[i][0])
    return S

def marginal_density_GMM (S):
    return scipy.special.logsumexp(S, axis = 0)


def log_likelihood_GMM(logmarg, X):
    return numpy.sum(logmarg)/X.shape[1]

def compute_posterior_GMM(X, gmm):
     return marginal_density_GMM(joint_log_density_GMM(logpdf_GMM(X, gmm),gmm))
 
def computeLogLikelihood(X, gmm):
    # SHOULD BE FIXED
    tempSum=numpy.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):
        tempSum[i,:]=numpy.log(gmm[i][0])+multivariateGaussian.logpdf_GAU_ND(X, gmm[i][1], gmm[i][2])
    return scipy.special.logsumexp(tempSum, axis=0)

def EMalgorithm(X, gmm, delta=1e-6):
    flag = True
    while(flag):
        # Given the training set and the initial model parameters, compute
        # log marginal densities and sub-class conditional densities
        S = joint_log_density_GMM(logpdf_GMM(X, gmm), gmm)  #matrix containing component conditional densities +
        logmarg= marginal_density_GMM(joint_log_density_GMM(logpdf_GMM(X, gmm), gmm)) #Compute the marginal densities

        loglikelihood1 = log_likelihood_GMM(logmarg, X)

        P = Estep(logmarg, S) #compute the posterior distribution for each sample

        #(w, mu, cov) = Mstep(X, S, posterior) #obtained model paramenter
        #for g in range(len(gmm)):
        #    gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])  # Update the model parameters in gmm
        
        # Compute the new log densities and the new sub-class conditional densities
        #logmarg= marginal_density_GMM(joint_log_density_GMM(logpdf_GMM(X, gmm), gmm) )                                                                            #aggiustare
        #loglikelihood2 = log_likelihood_GMM(logmarg, X)
        
        newGmm = []
        for i in range(len(gmm)):
            gamma = P[i, :]
            Z = gamma.sum()
            F = (generics.mrow(gamma) * X).sum(1)
            S = numpy.dot(X, (generics.mrow(gamma) * X).T)
            w = Z / X.shape[1]
            mu = generics.mcol(F / Z)
            sigma = S / Z - numpy.dot(mu, mu.T)
            U, s, _ = numpy.linalg.svd(sigma)
            s[s < 0.01] = 0.01
            sigma = numpy.dot(U, generics.mcol(s) * U.T)
            newGmm.append((w, mu, sigma))
        gmm = newGmm
        
        logmarg= marginal_density_GMM(joint_log_density_GMM(logpdf_GMM(X, gmm), gmm) )                                                                            #aggiustare
        loglikelihood2 = log_likelihood_GMM(logmarg, X)
        
        if (loglikelihood2-loglikelihood1 < delta):
            flag = False
        if (loglikelihood2-loglikelihood1 < 0):
            print("LOG-LIKELIHOOD IS NOT INCREASING: it is very likely to be incorrect")
    return gmm

def Estep(logdens, S):
    #E-step: compute the posterior probability for each component of the GMM for each sample, using
    #an estimate (Mt;St;wt) of the model parameters. These quantities are also called responsibilities
    return numpy.exp(S-logdens.reshape(1, logdens.size))

def Mstep(X, S, posterior):
    #M-step: Update the model parameters using statistics..
    Zg = numpy.sum(posterior, axis=1)  # 3
    Fg = numpy.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        tempSum = numpy.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * X[:, i]
        Fg[:, g] = tempSum
    Sg = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        tempSum = numpy.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * numpy.dot(X[:, i].reshape(
                (X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        Sg[g] = tempSum
    mu = Fg / Zg
    prodmu = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = numpy.dot(mu[:, g].reshape((X.shape[0], 1)),
                           mu[:, g].reshape((1, X.shape[0])))
    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
    for g in range(S.shape[0]):        
        cov[g] = generics.constrainSigma(cov[g])
    w = Zg/numpy.sum(Zg)
    return (w, mu, cov)

def split(GMM, alpha = 0.1):
    #we are displacing the new components along the direction of maximum variance, using a
    #step that is proportional to the standard deviation
    #we produce a 2G-components GMM from a G-components GMM. The 2G components GMM can be used as initial GMM for the EM algorithm.
    size = len(GMM)
    splittedGMM = []
    for i in range(size):
        U, s, Vh = numpy.linalg.svd(GMM[i][2])
        # compute displacement vector
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]+d, GMM[i][2]))
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]-d, GMM[i][2]))
    return splittedGMM

def LBGalgorithm(GMM, X, iterations):
    GMM = EMalgorithm(X, GMM)
    for i in range(iterations):
        GMM = split(GMM)
        GMM = EMalgorithm(X, GMM)
    return GMM

def DiagLBGalgorithm(GMM, X, iterations):
    GMM = DiagEMalgorithm(X, GMM)
    for i in range(iterations):
        GMM = split(GMM)
        GMM = DiagEMalgorithm(X, GMM)
    return GMM


def TiedLBGalgorithm(GMM, X, iterations):
    GMM = TiedEMalgorithm(X, GMM)
    for i in range(iterations):
        GMM = split(GMM)
        GMM = TiedEMalgorithm(X, GMM)
    return GMM

def DiagEMalgorithm(X, gmm, delta=10**(-6)):
    flag = True
    while(flag):
        # Given the training set and the initial model parameters, compute
        # log marginal densities and sub-class conditional densities
        S = joint_log_density_GMM(logpdf_GMM(X, gmm), gmm)
        logmarg= marginal_density_GMM(joint_log_density_GMM(logpdf_GMM(X, gmm), gmm) )
        # Compute the AVERAGE loglikelihood, by summing all the log densities and
        # dividing by the number of samples (it's as if we're computing a mean)
        loglikelihood1 = log_likelihood_GMM(logmarg, X)
        # ------ E-step ----------
        posterior = Estep(logmarg, S)
        # ------ M-step ----------
        #(w, mu, cov) = DiagMstep(X, S, posterior)
        #for g in range(len(gmm)):
            # Update the model parameters that are in gmm
        #    gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])

        newGmm = []
        for i in range(len(gmm)):
            gamma = posterior[i, :]
            Z = gamma.sum()
            F = (generics.mrow(gamma) * X).sum(1)
            S = numpy.dot(X, (generics.mrow(gamma) * X).T)
            w = Z / X.shape[1]
            mu = generics.mcol(F / Z)
            sigma = S/Z - numpy.dot(mu, mu.T)
            sigma *= numpy.eye(sigma.shape[0]) # Diag cov
            U, s, _ = numpy.linalg.svd(sigma)
            s[s < 0.01] = 0.01
            sigma = numpy.dot(U, generics.mcol(s) * U.T)
            newGmm.append((w, mu, sigma))
        gmm = newGmm

        # Compute the new log densities and the new sub-class conditional densities
        
        logmarg= marginal_density_GMM(joint_log_density_GMM(logpdf_GMM(X, gmm), gmm) )
        loglikelihood2 = log_likelihood_GMM(logmarg, X)
        if (loglikelihood2-loglikelihood1 < delta):
            flag = False
        if (loglikelihood2-loglikelihood1 < 0):
            print("ERROR, LOG-LIKELIHOOD IS NOT INCREASING")
    return gmm

def DiagMstep(X, S, posterior):
    Zg = numpy.sum(posterior, axis=1)  # 3
    Fg = numpy.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        tempSum = numpy.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * X[:, i]
        Fg[:, g] = tempSum
    Sg = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        tempSum = numpy.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * numpy.dot(X[:, i].reshape(
                (X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        Sg[g] = tempSum
    mu = Fg / Zg
    prodmu = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = numpy.dot(mu[:, g].reshape((X.shape[0], 1)),
                           mu[:, g].reshape((1, X.shape[0])))
    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
    for g in range(S.shape[0]):
       
        cov[g] = generics.constrainSigma(cov[g] * numpy.eye(cov[g].shape[0]))
    w = Zg/numpy.sum(Zg)
    return (w, mu, cov)


def TiedEMalgorithm(X, gmm, delta=10**(-6)):
    flag = True
    while(flag):

        S = joint_log_density_GMM(logpdf_GMM(X, gmm), gmm)
        logmarg= marginal_density_GMM(joint_log_density_GMM(logpdf_GMM(X, gmm), gmm) )

        loglikelihood1 = log_likelihood_GMM(logmarg, X)

        posterior = Estep(logmarg, S)

        #(w, mu, cov) = TiedMstep(X, S, posterior)
        #for g in range(len(gmm)):

        #    gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
        #


        #

        newGmm = []
        sigmaTied = numpy.zeros((X.shape[0], X.shape[0]))
        for i in range(len(gmm)):
            gamma = posterior[i, :]
            Z = gamma.sum()
            F = (generics.mrow(gamma) * X).sum(1)
            S = numpy.dot(X, (generics.mrow(gamma) * X).T)
            w = Z/ X.shape[1]
            mu = generics.mcol(F/Z)
            sigma = S/Z - numpy.dot(mu, mu.T)
            sigmaTied += Z * sigma
            newGmm.append((w, mu))   
        gmm = newGmm
        sigmaTied /= X.shape[1]
        U, s, _ = numpy.linalg.svd(sigmaTied)
        s[s<0.01] = 0.01
        sigmaTied = numpy.dot(U, generics.mcol(s) * U.T)
        
        newGmm = []
        for i in range(len(gmm)):
            (w, mu) = gmm[i]
            newGmm.append((w, mu, sigmaTied))
            
        gmm = newGmm


        logmarg= marginal_density_GMM(joint_log_density_GMM(logpdf_GMM(X, gmm), gmm) )                                                                            #aggiustare
        loglikelihood2 = log_likelihood_GMM(logmarg, X)
        if (loglikelihood2-loglikelihood1 < delta):
            flag = False
        if (loglikelihood2-loglikelihood1 < 0):
            print("LOG NOT INCREASING")
    return gmm

def TiedMstep(X, S, posterior):
    Zg = numpy.sum(posterior, axis=1)  
    Fg = numpy.zeros((X.shape[0], S.shape[0])) 
    for g in range(S.shape[0]):
        tempSum = numpy.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * X[:, i]
        Fg[:, g] = tempSum
    Sg = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        tempSum = numpy.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * numpy.dot(X[:, i].reshape(
                (X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        Sg[g] = tempSum
    mu = Fg / Zg
    prodmu = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = numpy.dot(mu[:, g].reshape((X.shape[0], 1)),
                           mu[:, g].reshape((1, X.shape[0])))
    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
    tsum = numpy.zeros((cov.shape[1], cov.shape[2]))
    for g in range(S.shape[0]):
        tsum += Zg[g]*cov[g]
    for g in range(S.shape[0]):
        cov[g] = generics.constrainSigma(1/X.shape[1] * tsum)
    w = Zg/numpy.sum(Zg)
    return (w, mu, cov)