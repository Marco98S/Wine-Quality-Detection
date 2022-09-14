import numpy
import lib.LogisticRegression as LR
from lib.PCA import PCA
import lib.dcfFun
from scipy.stats import norm,rankdata
import lib.plotFunctions as plot
import lib.generics as generics

def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:8]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                pulsar = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(pulsar)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


def normalize_zscore(D, mu=[], sigma=[]):
    if mu == [] or sigma == []:
        mu = numpy.mean(D, axis=1)
        sigma = numpy.std(D, axis=1)
    ZD = D
    ZD = ZD - mcol(mu)
    ZD = ZD / mcol(sigma)
    return ZD, mu, sigma

def gaussianization(DTR, DTE = None):
    #return numpy.array([norm.ppf((rankdata(x_i) - 0.5) / len(x_i)) for x_i in D])
    rankDTR = numpy.zeros(DTR.shape)
    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[1]):
            rankDTR[i][j] = (DTR[i] < DTR[i][j]).sum()
    rankDTR = (rankDTR+1) / (DTR.shape[1]+2)
    
    if(DTE is not None):
        rankDTE = numpy.zeros(DTE.shape)
        for i in range(DTE.shape[0]):
            for j in range(DTE.shape[1]):
                rankDTE[i][j] = (DTR[i] < DTE[i][j]).sum()
        rankDTE = (rankDTE+1) / (DTR.shape[1]+2)
        return norm.ppf(rankDTR), norm.ppf(rankDTE)
    return norm.ppf(rankDTR)

def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return (v.reshape(1, v.size))

def centeredData(D):
    return D - mcol(D.mean(1))

def computeAccuracy(predictedLabels, actualLabels):
    numberOfCorrectPredictions = numpy.array(predictedLabels == actualLabels).sum()
    accuracy = numberOfCorrectPredictions/actualLabels.size*100
    return accuracy

def computeErrorRate(predictedLabels, actualLabels):
    accuracy = computeAccuracy(predictedLabels, actualLabels)
    errorRate = 100-accuracy
    return errorRate

def confusionMatrix(Pred, Labels, K):
    # Initialize matrix of K x K zeros
    Conf = numpy.zeros((K, K)).astype(int)
    # We're computing the confusion
    # matrix which "counts" how many times we get prediction i when the actual
    # label is j.
    for i in range(K):
        for j in range(K):
            Conf[i,j] = ((Pred == i) * (Labels == j)).sum()
    return Conf

def Ksplit(D, L, seed=0, K=3):
    folds = []
    labels = []
    numberOfSamplesInFold = int(D.shape[1]/K)
    # Generate a random seed
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    for i in range(K):
        folds.append(D[:,idx[(i*numberOfSamplesInFold): ((i+1)*(numberOfSamplesInFold))]])
        labels.append(L[idx[(i*numberOfSamplesInFold): ((i+1)*(numberOfSamplesInFold))]])
    return folds, labels

def Kfold(D, L, model, mPCA, gauss=True, K=3, prior=0.5, ACT=False,calibrate=False):
    folds, labels = Ksplit(D, L, seed=0, K=K)
    
    orderedLabels = []
    scores = []
    for i in range(K):
        trainingSet = []
        labelsOfTrainingSet = []
        for j in range(K):
            if j!=i:
                trainingSet.append(folds[j])
                labelsOfTrainingSet.append(labels[j])
        evaluationSet = folds[i]
        orderedLabels.append(labels[i])
        trainingSet=numpy.hstack(trainingSet)
        labelsOfTrainingSet=numpy.hstack(labelsOfTrainingSet)

        trainingSet, mu, sigma = normalize_zscore(trainingSet)
        evaluationSet, mu, sigma = normalize_zscore(evaluationSet, mu, sigma)
        
        if(gauss):
            trainingSet, evaluationSet = gaussianization(trainingSet, evaluationSet)
        
        if(mPCA != 0):
            trainingSet, P = PCA(trainingSet, mPCA)
            evaluationSet = numpy.dot(P.T, evaluationSet)
        
        model.train(trainingSet, labelsOfTrainingSet)
        scores.append(model.predictAndGetScores(evaluationSet))
    scores=numpy.hstack(scores)
    orderedLabels=numpy.hstack(orderedLabels)
    labels = numpy.hstack(labels)
    minimum_DCF = lib.dcfFun.minimum_detection_costs(scores, orderedLabels, prior, 1, 1)
    if(calibrate):
        act_dcf = calibrateScores(scores, orderedLabels, 1e-5, prior)
        return minimum_DCF, act_dcf
    if(not ACT):
        return minimum_DCF
    return minimum_DCF, lib.dcfFun.compute_actual_DCF(scores, orderedLabels, prior, 1, 1)




def calibrateScores(s, L, lambd, prior=0.5, evalu=False):
    s=mrow(s)
    
    #split score with single fold
    (scoresTR, labelsTR), (scoresTE, labelsTE) = split_db_2to1(s, L)
    
    lr = LR.LogisticRegression(lambd, prior)
    lr.train(scoresTR, labelsTR)
    alpha = lr.x[0]
    betafirst = lr.x[1]
    calibScores = alpha*scoresTE+betafirst-numpy.log(prior/(1-prior))
    if(evalu):
        return calibScores.flatten(), labelsTE
    return lib.dcfFun.compute_actual_DCF(calibScores, labelsTE, prior, 1, 1)

def split_db_2to1(D, L):
    nTrain = int(D.shape[1] * 2./3.)
    numpy.random.seed(0)
    index = numpy.random.permutation(D.shape[1])
    iTrain = index[0:nTrain]
    iTest = index[nTrain:]
    DTR = D[:, iTrain]
    DTE = D[:, iTest]
    LTR = L[iTrain]
    LTE = L[iTest]
    return (DTR, LTR), (DTE, LTE)


def constrainSigma(sigma, psi = 0.01):

    U, s, Vh = numpy.linalg.svd(sigma)
    s[s < psi] = psi
    sigma = numpy.dot(U, mcol(s)*U.T)
    return sigma


def computeOptBayesDecision(pi1, Cfn, Cfp, llrs, labels, t):
    predictedLabels = (llrs > t).astype(int)
    # Compute the confusion matrix
    m = confusionMatrix(predictedLabels, labels, 2)
    return m

def computeFPRTPR(pi1, Cfn, Cfp, confMatrix):
    # Compute FNR and FPR
    FNR = confMatrix[0][1]/(confMatrix[0][1]+confMatrix[1][1])
    TPR = 1-FNR
    FPR = confMatrix[1][0]/(confMatrix[0][0]+confMatrix[1][0])
    return (FPR, TPR)