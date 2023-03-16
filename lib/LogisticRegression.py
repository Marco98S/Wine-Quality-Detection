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
    
class QuadLR:
    
    def __init__(self, l, pi_T):
        self.l = l
        self.pi_T = pi_T
    
    
    def train(self, D, L):
        
        phi = mapToFeatureSpace(D)
        
        self.x, self.f, self.d = scipy.optimize.fmin_l_bfgs_b(lib.LogRegFunctions.logreg_obj, np.zeros(
            phi.shape[0] + 1), args=(phi, L, self.l, self.pi_T), approx_grad=True)
        #w, b = self.x[0:-1], self.x[-1]
        
    def predictAndGetScores(self, X):
        X = mapToFeatureSpace(X)
        scores = np.dot(self.x[0:-1], X) + self.x[-1]
        return scores
    
def mapToFeatureSpace(D):
        phi = np.zeros([D.shape[0]**2+D.shape[0], D.shape[1]])
        for index in range(D.shape[1]):
            x = D[:, index].reshape(D.shape[0], 1)
            # phi = [vec(x*x^T), x]^T
            phi[:, index] = np.concatenate((np.dot(x, x.T).reshape(x.shape[0]**2, 1), x)).reshape(phi.shape[0],)
        return phi