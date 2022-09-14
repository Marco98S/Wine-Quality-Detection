import numpy as np
import scipy.optimize
import lib.LogRegFunctions

class LogisticRegression:

    def __init__(self, l, pi_T):
        self.l = l
        self.pi_T = pi_T


    def train(self, D, L):
        D0 = D[:, L == 0]
        D1 = D[:, L == 1]
        self.nF = D0.shape[1]
        self.nT = D1.shape[1]
        self.x, self.f, self.d = scipy.optimize.fmin_l_bfgs_b(lib.LogRegFunctions.logreg_obj, np.zeros(
            D.shape[0] + 1), args=(D, L, self.l, self.pi_T), approx_grad=True)      

    def predict(self, X):
        scores = np.dot(self.x[0:-1], X) + self.x[-1]
        predictedLabels = (scores>0).astype(int)
        return predictedLabels
    
    def predictAndGetScores(self, X):
        #scores = np.dot(self.x[0:-1], X) + self.x[-1] - np.log(self.nT/self.nF)
        scores = np.dot(self.x[0:-1], X) + self.x[-1]
        return scores