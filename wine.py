import numpy
from lib.GMMClassifiers import GMM, GMMDiag, GMMTiedCov
from lib.GaussianClassifiers import GaussianClassifier, GaussianClassifierNaiveBayes
from lib.GaussianClassifiers import GaussianClassifierTiedCov
import lib.generics as generics
import lib.PCA  as PCA
import lib.LogisticRegression as LR
import lib.plotFunctions as plot
import lib.SVM as SVM
from lib.generics import load
from lib.dcfFun import minimum_detection_costs, compute_actual_DCF

priors = [0.5, 0.1, 0.9]
MVGmodels = {"MVG Full" : GaussianClassifier(), "MVG Diag":  GaussianClassifierNaiveBayes(), "MVG Tied" : GaussianClassifierTiedCov()}
PCAoptions = {"No PCA": 0 , "PCA 8": 8, "PCA 7": 7,"PCA 6": 6, "PCA 5" : 5}
#PCAoptions = {"No PCA": 0 , "PCA 8": 8, "PCA 7": 7,"PCA 6": 6}
#PCAoptions = {"NoPCA": 0 , "PCA8": 8, "PCA7": 7}
#PCAoptions = {"NoPCA": 0 , "PCA7": 7}

def featureAnalysis(D,L):
    plot.plot_hist(DTR, L, "raw")
    NDTR, _, _ = generics.normalize_zscore(D)
    plot.plot_hist(NDTR, L, "z-score")
    GDTR = generics.gaussianization(D)
    plot.plot_hist(GDTR, L, "Gauss")

    plot.heatmap(NDTR, "raw", "Blues")
    plot.heatmap(NDTR[:, L==1], "raw_pos", "Greens")
    plot.heatmap(NDTR[:, L==0], "raw_neg", "Reds")


if __name__ == "__main__":
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")

    featureAnalysis(DTR, LTR)
    