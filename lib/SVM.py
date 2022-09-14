import scipy.optimize 
import numpy

from lib.generics import mcol, mrow

class SVM ():
    
    def __init__(self, option, pi_T=0, gamma=1.0, c=0, C=1.0, d=2, K=1.0):
        self.option = option
        self.pi_T = pi_T
        self.c = c
        self.d = d
        self.gamma = gamma
        self.C = C
        self.K = K
    
    def train (self, DTR, LTR):
        
        self.DTR = DTR
        self.LTR = LTR

        b = []
        if(self.pi_T == 0):
            b = [(0,self.C)] * DTR.shape[1]
        else:
            C1 = self.C * self.pi_T / (DTR[:,LTR == 1].shape[1]/DTR.shape[1])
            C0 = self.C * (1-self.pi_T) / (DTR[:,LTR == 0].shape[1]/DTR.shape[1])
            for i in range(DTR.shape[1]):
                if LTR[i]== 1:
                    b.append ((0,C1))
                elif LTR[i]== 0:
                    b.append ((0,C0))
        
        if (self.option == 'linear'):
            self.w = modifiedDualFormulation(DTR, LTR, b, self.C, self.K)
            
        elif (self.option == 'polynomial'):
            self.x = Polykernel(DTR, LTR, b, self.K, self.C, self.d, self.c)
        elif (self.option == 'RBF'):
            self.x, self.Z = RBFkernel(DTR, LTR, b, self.gamma, self.K, self.C) 
            
    
    def predict (self, DTE):
        
        if (self.option == 'linear'):

            DTE = numpy.vstack([DTE, numpy.zeros(DTE.shape[1])+self.K])
            S = numpy.dot(self.w.T, DTE)
            LP = 1*(S > 0)
            LP[LP == 0] = -1
            return LP
        elif (self.option == 'polynomial'):

            S = numpy.sum(
                numpy.dot((self.x*self.LTR).reshape(1, self.DTR.shape[1]), (numpy.dot(self.DTR.T, DTE)+self.c)**self.d+ self.K), axis=0)
            LP = 1*(S > 0)
            LP[LP == 0] = -1
            return LP
        elif (self.option == 'RBF'):
            
            s = numpy.zeros(DTE.shape[1])
            
            Dist = mcol((self.DTR**2).sum(0)) + mrow((DTE**2).sum(0)) - 2 * numpy.dot(self.DTR.T, DTE)
            k = numpy.exp(-self.gamma*Dist)  + self.K**2 
            s = numpy.dot((mcol(self.x) * mcol(self.Z)).T,k)
            s.sum(0)
            s=s.ravel()

            return s
    
    def predictAndGetScores(self, DTE):
        if (self.option == 'linear'):

             DTE = numpy.vstack([DTE, numpy.zeros(DTE.shape[1])+self.K])
             S = numpy.dot(self.w.T, DTE)
             return S
        if (self.option == 'polynomial'):

            S = numpy.sum(
                numpy.dot((self.x*self.LTR).reshape(1, self.DTR.shape[1]), (numpy.dot(self.DTR.T, DTE)+self.c)**self.d+ self.K), axis=0)
            return S
        if (self.option == 'RBF'):
            s = numpy.zeros(DTE.shape[1])

            Dist = mcol((self.DTR**2).sum(0)) + mrow((DTE**2).sum(0)) - 2 * numpy.dot(self.DTR.T, DTE)
            k = numpy.exp(-self.gamma*Dist)  + self.K**2 
            s = numpy.dot((mcol(self.x) * mcol(self.Z)).T,k)
            s.sum(0)
            s=s.ravel()
            
            predLabels = numpy.int32(s>0)
            return s, predLabels


def JDual(H, alpha):
    Ha = numpy.dot(H, mcol(alpha))
    aHa = numpy.dot(mrow(alpha), Ha)
    a1 = alpha.sum()
    return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

def LDual(alpha, H):
    loss , grad = JDual(H, alpha)
    return -loss, -grad

#def JPrimal(w):
    #DTREXT = numpy.vstack([DTR, numpy.ones((1,DTR.shape[1]))*K])
    #S = numpy.dot(mrow(w), DTREXT)
    #loss = numpy.maximum(numpy.zeros(S.shape), 1-Z*S).sum()
    #return 0.5 * numpy.linalg.norm(w)**2 + C * loss



def modifiedDualFormulation(DTR, LTR, b, C, K):
    # Compute the D matrix for the extended training set
    
    row = numpy.zeros(DTR.shape[1])+K
    D = numpy.vstack([DTR, row])

    # Compute the H matrix 
    Gij = numpy.dot(D.T, D)
    zizj = numpy.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*Gij
    
    def computeDualLoss(alpha, H):
        grad = numpy.dot(H, alpha) - numpy.ones(H.shape[1])
        return ((1/2)*numpy.dot(numpy.dot(alpha.T, H), alpha)-numpy.dot(alpha.T, numpy.ones(H.shape[1])), grad)

    (x, f, d) = scipy.optimize .fmin_l_bfgs_b(computeDualLoss,
                                    numpy.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=0, factr=1e7, maxiter = 100000,
            maxfun = 100000)
    return numpy.sum((x*LTR).reshape(1, DTR.shape[1])*D, axis=1)


def Polykernel(DTR, LTR, b, K, C, d, c):
    # Compute the H matrix
    kernelFunction = (numpy.dot(DTR.T, DTR)+c)**d+ K**2
    zizj = numpy.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction

    (x, f, data) = scipy.optimize.fmin_l_bfgs_b(LDual,
                                    numpy.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=0, factr=1.0)
    return x


def RBF(x1, x2, gamma, K):
    return numpy.exp(-gamma*(numpy.linalg.norm(x1-x2)**2))+K**2

def RBFkernel(DT, LT, b, gamma,  K, C):
    Z = numpy.zeros(LT.shape)
    Z[LT == 1] = 1
    Z[LT == 0] = -1
    
    dist = mcol((DT**2).sum(0)) + mrow((DT**2).sum(0)) - 2*numpy.dot(DT.T, DT)
    kernel = numpy.exp(-gamma * dist) + K**2
    
    H = mcol(Z) * mrow(Z) * kernel
    
    (x, f, data) = scipy.optimize.fmin_l_bfgs_b(LDual,
                                    numpy.zeros(DT.shape[1]), args=(H,), bounds=b, iprint=0, factr=1.0)
    return x,Z