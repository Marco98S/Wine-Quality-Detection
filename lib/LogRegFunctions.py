import numpy as np

def Jgradrebalanced(w, b, DTR, LTR, lambd, prior):
    norm = lambd/2*(np.linalg.norm(w)**2)
    sumTermTrueClass = 0
    sumTermFalseClass = 0
    for i in range(DTR.shape[1]):
        argexpnegative = -np.dot(w.T, DTR[:, i])-b
        flagArgExpNegative = False
        argexppositive = np.dot(w.T, DTR[:, i])+b
        flagArgExpPositive = False
        if (argexpnegative>709):
            flagArgExpNegative=True
        if (argexppositive>709):
            flagArgExpPositive=True
        if LTR[i]==1:
            if (flagArgExpNegative==True):
                sumTermTrueClass += argexpnegative
            else:
                sumTermTrueClass += np.log1p(np.exp(-np.dot(w.T, DTR[:, i])-b))
        else:
            if (flagArgExpPositive==True):
                sumTermFalseClass+=argexppositive
            else:
                sumTermFalseClass += np.log1p(np.exp(np.dot(w.T, DTR[:, i])+b))
    j = norm + (prior/DTR[:, LTR==1].shape[1])*sumTermTrueClass + ((1-prior)/DTR[:, LTR==0].shape[1])*sumTermFalseClass
    return j

def logreg_obj(v, DTR, LTR, l, pi_T=0.5):
    w, b = v[0:-1], v[-1]
    #return Jgradrebalanced(w, b, DTR, LTR, l, pi_T)
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    S0 = np.dot(w.T, D0) + b
    S1 = np.dot(w.T, D1) + b
    crossEntropy = pi_T * np.logaddexp(0, -S1).mean()
    crossEntropy += (1-pi_T) * np.logaddexp(0, S0).mean()
    return  0.5*l * np.linalg.norm(w)**2 + crossEntropy
