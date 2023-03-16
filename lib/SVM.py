import numpy
import lib.generics as generics
import scipy.optimize


def dual_wrapper(D, H, bounds):
    def LDual(alpha, H):
        Ha = numpy.dot(H, generics.mcol(alpha))
        aHa = numpy.dot(generics.mrow(alpha), Ha)
        a1 = alpha.sum()
        return 0.5 * aHa.ravel() - a1,  Ha.ravel() - numpy.ones(alpha.size)

    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(D.shape[1]),
        args=(H,),
        bounds=bounds
    )

    return alphaStar


class SVM:

    def __init__(self, option, pi=0, K=1, C=0, c=0, d=2, gamma=0):
        self.option = option
        self.K = K
        self.C = C
        self.c = c
        self.d = d
        self.gamma = gamma
        self.pi = pi
        
    def train(self, D, L):
        
        self.DTR = D
        self.LTR = L
        self.Z = numpy.zeros(L.shape)
        self.Z[L == 1] = 1
        self.Z[L == 0] = -1
        

        if(self.pi != 0):
            C1 = (self.C * self.pi) / (D[:, L == 1].shape[1] / D.shape[1])
            C0 = (self.C * (1 - self.pi)) / (D[:, L == 0].shape[1] / D.shape[1])
            self.bounds = [((0, C0) if x == 0 else (0, C1)) for x in L.tolist()]
        else:
            self.bounds = [(0, self.C)] * D.shape[1]

        if(self.option == 'linear'):
            DTRT = numpy.vstack([D, numpy.ones(D.shape[1]) * self.K])
            H = numpy.dot(DTRT.T, DTRT)
            H = numpy.dot(generics.mcol(self.Z), generics.mrow(self.Z)) * H
            alphaStar = dual_wrapper(D, H, self.bounds)
            self.w = numpy.dot(DTRT, generics.mcol(alphaStar) * generics.mcol(self.Z)).sum(axis = 1)
        if(self.option == 'RBF'):
            Dist = generics.mcol((D ** 2).sum(0)) + generics.mrow((D ** 2).sum(0)) - 2 * numpy.dot(D.T, D)
            kernel = numpy.exp(-self.gamma * Dist) + (self.K ** 2)
            H = numpy.dot(generics.mcol(self.Z), generics.mrow(self.Z)) * kernel
            self.w = dual_wrapper(D, H, self.bounds)
        if(self.option == 'poly'):
            
            kernel = ((numpy.dot(D.T, D) + self.c) ** self.d) + (self.K ** 2)
            H = numpy.dot(generics.mcol(self.Z), generics.mrow(self.Z)) * kernel
            self.w = dual_wrapper(D, H, self.bounds)
        return self


    def predictAndGetScores(self, D):
        if(self.option == 'linear'):
            DTET = numpy.vstack([D, numpy.ones(D.shape[1]) * self.K])
            return numpy.dot(self.w.T, DTET)
        if(self.option == 'RBF'):
            Dist = generics.mcol((self.DTR ** 2).sum(0)) + generics.mrow((D ** 2).sum(0)) - 2 * numpy.dot(self.DTR.T, D)
            kernel = numpy.exp(-self.gamma * Dist) + (self.K ** 2)
            return numpy.dot(self.w * self.Z, kernel)
        if(self.option == 'poly'):
            kernel = ((numpy.dot(self.DTR.T, D) + self.c) ** self.d) + (self.K ** 2)
            return numpy.dot(self.w * self.Z, kernel)