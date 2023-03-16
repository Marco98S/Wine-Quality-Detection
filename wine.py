import numpy
from lib.GMMClassifiers import GMM#, GMMDiag, GMMTiedCov
from lib.GaussianClassifiers import GaussianClassifier, GaussianClassifierNaiveBayes
from lib.GaussianClassifiers import GaussianClassifierTiedCov, GaussianClassifierTiedDiagCov
import lib.generics as generics
import lib.PCA  as PCA
import lib.LogisticRegression as LR
import lib.plotFunctions as plot
import lib.SVM as SVM
from lib.generics import load
from lib.dcfFun import minimum_detection_costs, compute_actual_DCF

priors = [0.5, 0.1, 0.9]
MVGmodels = {"MVG Full" : GaussianClassifier(), "MVG Diag":  GaussianClassifierNaiveBayes(), "MVG Tied" : GaussianClassifierTiedCov(), "MVG TiedDiag" :  GaussianClassifierTiedDiagCov()}
#PCAoptions = {"No PCA": 0 , "PCA 10" : 10,"PCA 9" : 9, "PCA 8": 8, "PCA 7": 7}
PCAoptions = {"NoPCA": 0}
#PCAoptions = {"PCA9": 9 , "PCA8": 8}

def featureAnalysis(D,L):
    plot.plot_hist(DTR, L, "raw")
    NDTR, _, _ = generics.normalize_zscore(D)
    plot.plot_hist(NDTR, L, "z-score")
    GDTR = generics.gaussianization(D)
    plot.plot_hist(GDTR, L, "Gauss")

    plot.heatmap(NDTR, "raw", "Blues")
    plot.heatmap(NDTR[:, L==1], "raw_pos", "Greens")
    plot.heatmap(NDTR[:, L==0], "raw_neg", "Reds")

def MVG(D, L):
    for gauss in [True, False]:
        if(gauss): print("Gauss Features")
        else: print("Raw features")
        for o in PCAoptions:
            print("____________________________________")
            print("  %s" %o)
            for m in MVGmodels:
                for i in range(3):
                    min_dcf = generics.Kfold(D, L, MVGmodels[m], PCAoptions[o], gauss=gauss, prior=priors[i])
                    print("    %s with prior=%.1f:  %.3f" %(m, priors[i], min_dcf))
                print("")


def compareModelLR(D, L):
    lambdas=numpy.logspace(-6, 2, num=25)
    min_dcf = []
    
    for gauss in [False, True]:
        if(gauss): print("Gauss Features")
        else: print("Raw features")
        for o in PCAoptions:
            print("____________________________________")
            print("  %s" %o)
            min_dcf = []
            for i in range(len(priors)):
                print("    Working on application with prior:", priors[i])
                for l in lambdas:
                    lr = LR.LogisticRegression(l, 0.5)
                    m = generics.Kfold(D, L, lr, PCAoptions[o], gauss=gauss, prior=priors[i])
                    min_dcf.append(m)
                    print("      For lambda", l, "the minDCF is", m)
            plt = plot.plotDCF(lambdas, min_dcf, "λ")
            fileName = ""
            if(gauss): 
                fileName = "plots/LR/LR" + o.strip() + ".pdf"
            else: 
                fileName = "plots/LR/RawLR" + o.strip() + ".pdf"
            print(fileName)
            plt.savefig(fileName)
            print("\n")

def logisticRegression(D, L):
    #lgauss = 1e-2
    #lraw = 0
    l = 1e-2
    
    for gauss in [True, False]:
        if(gauss): 
            print("Gauss Features")
            #l = lgauss
        else:
            #l = lraw
            print("Raw features")
        for o in PCAoptions:
            print("____________________________________")
            print("  %s" %o)
            for pi_t in priors:
                lr = LR.LogisticRegression(l, pi_t)
                for i in range(len(priors)):
                    min_dcf = generics.Kfold(D, L, lr, PCAoptions[o], gauss=gauss, prior=priors[i])
                    print("    LR with pI_T = %.1f and prior = %.1f:  %.3f" %(pi_t, priors[i], min_dcf))

                    
def hyperparamsQuadLR(D, L):
    lambdas=numpy.logspace(-6, 2, num=25)
    min_dcf = []
    
    for gauss in [False, True]:
        if(gauss): print("Gauss Features")
        else: print("Raw features")
        for o in PCAoptions:
            print("____________________________________")
            print("  %s" %o)
            min_dcf = []
            for i in range(len(priors)):
                print("    Working on application with prior:", priors[i])
                for l in lambdas:
                    lr = LR.QuadLR(l, 0.5)
                    m = generics.Kfold(D, L, lr, PCAoptions[o], gauss=gauss, prior=priors[i])
                    min_dcf.append(m)
                    print("      For lambda", l, "the minDCF is", m)
            plt = plot.plotDCF(lambdas, min_dcf, "λ")
            fileName = ""
            if(gauss): 
                fileName = "plots/LR/QuadLR" + o.strip() + ".pdf"
            else: 
                fileName = "plots/LR/QuadRawLR" + o.strip() + ".pdf"
            print(fileName)
            plt.savefig(fileName)
            print("\n")
        

def QuadlogisticRegression(D, L):
    lgauss = 1e-2
    lraw = 0
    l = 0
    for gauss in [False, True]:
        if(gauss): 
            print("Gauss Features")
            #l = lgauss
        else:
            #l = lraw
            print("Raw features")
        for o in PCAoptions:
            print("____________________________________")
            print("  %s" %o)
            for pi_t in priors:
                lr = LR.QuadLR(l, pi_t)
                for i in range(len(priors)):
                    min_dcf = generics.Kfold(D, L, lr, PCAoptions[o], gauss=gauss, prior=priors[i])
                    print("    LR with pI_T = %.1f and prior = %.1f:  %.3f" %(pi_t, priors[i], min_dcf))        

def compareModelSVM(D,L):
    C=numpy.logspace(-3, 1, num=15)
    for gauss in [True, False]:
        if(gauss): print("Gauss Features")
        else: print("Raw features")
        for o in PCAoptions:
            for pi_T in [0.5,0]:
                min_cdf = []
                for p in priors:
                    print("")
                    print("Working on application with prior:", p)
                    for value in C:
                        svm = SVM.SVM("linear", pi=pi_T, C=value)
                        #(self, type='linear', pi=0, K=1, C=0, c=0, d=2, gamma=0):
                        #self, D, L, type='linear', pi=0, balanced=False, K=1, C=0, c=0, d=2, gamma=0
                        #svm  = SVM.self, C, mode, pT, gamma=1, d=2, K=1
                        m = generics.Kfold(D, L, svm, PCAoptions[o], gauss=gauss, prior=p)
                        min_cdf.append(m)
                        print("For C=", value, "the minDCF is", m)
                    print("")
                plt = plot.plotDCF(C, min_cdf, "C",)
                fileName = ""
                bal = "UnBalanced"
                if(pi_T != 0):
                    bal = "Balanced"
                if(gauss): 
                    fileName = "plots/SVM/" + bal + "_" + "LinearSVM" + o.strip() + ".pdf"
                else: 
                    fileName = "plots/SVM/RAW" + bal + "_" + "LinearSVM" + o.strip() + ".pdf"
                plt.savefig(fileName)
                
def SVM_minDCF(D, L):
     
    C = 1e-1
    #C_Unbalanced = 1e-2
    #C_balanced = 7*1e-3
    #0,00010
    print("SVM")
    for gauss in [True, False]:
        if(gauss): print("Gauss Features")
        else: print("Raw features")
        for o in PCAoptions:
            print("____________________________________")
            print("  %s" %o)
            for pi_t in [0, 0.5, 0.1, 0.9]: #0=unbalanced
                if(pi_t == 0):
                    print("unbalanced")
                    #C = C_Unbalanced
                else: 
                    #C = C_balanced
                    print("balanced")
                svm = SVM.SVM("linear", pi = pi_t, C=C)
                for i in range(len(priors)):
                    min_dcf = generics.Kfold(D, L, svm, PCAoptions[o], gauss=gauss, prior=priors[i])
                    print("    Linear SVM with pI_T = %.1f and prior = %.1f:  %.3f" %(pi_t, priors[i], min_dcf))

def estimateRBFSVM(D,L):

    #gamma = [10**(-3),10**(-2),10**(-1)]
    #gamma = [-1, -0.7, -0.5]
    gamma = [10**(-1),10**(-0.7),10**(-0.5), 10**(-0.3), 10**(0.5)]
    C=numpy.logspace(-2, 2, num=15)

    
    min_dcf = []
    for gauss in [True, False]:
        if(gauss): print("Gauss Features")
        else: print("Raw features")
        for o in PCAoptions:
            for pi_T in [0.5,0]:
                min_dcf = []
                print("")
                print("Working on application with prior: -- 0.5")
                for g in gamma:
                    print("gamma: ", g)
                    print("")
                    for value in C:
                        #svm = SVM.SVM("RBF", pi_T= pi_T, gamma = g, C=value)
                        svm = SVM.SVM("RBF", pi=pi_T, C=value, gamma=g)
                        #(self, type='linear', pi=0, K=1, C=0, c=0, d=2, gamma=0):
                        m = generics.Kfold(D, L, svm, PCAoptions[o], gauss=gauss, prior=0.5)
                        min_dcf.append(m)
                        print("For C=", value, "the minDCF is", m)
                    print("")
                print("end gamma")
                plt = plot.plotDCFRBF(C, min_dcf, "C")
                fileName = ""
                bal = "UnBalanced"
                if(pi_T != 0):
                    bal = "Balanced"
                if(gauss): 
                    fileName = "plots/RBF/" + bal + "_" + "LinearSVM" + o.strip() + ".pdf" #+ "_g=" + g + ".pdf"
                else: 
                    fileName = "plots/RBF/RAW" + bal + "_" + "LinearSVM" + o.strip() + ".pdf" #+ "_g=" + g + ".pdf"
                plt.savefig(fileName)

def estimatePolySVM(D,L):
    print("polynomial")
    #C=numpy.logspace(-3, 3, num=15)
    C = numpy.logspace(-3, 3, num=15)
    c = [0, 1, 10, 30]
    
    min_dcf = []
    for gauss in [True, False]:
        if(gauss): print("Gauss Features")
        else: print("Raw features")
        for o in PCAoptions:
            for pi_T in [0.5,0]: #balanced and unbalanced
                min_dcf = []
                print("")
                print("Working on application with prior: -- 0.5")
                for _c in c:
                    print("c: ", _c)
                    print("")
                    for value in C:
                        #svm = SVM.SVM("polynomial", pi_T= pi_T, c = _c, C=value)
                        svm = SVM.SVM("poly", pi=pi_T, C=value, c = _c)
                        #(self, type='linear', pi=0, K=1, C=0, c=0, d=2, gamma=0):
                        m = generics.Kfold(D, L, svm, PCAoptions[o], gauss=gauss, prior=0.5)
                        min_dcf.append(m)
                        print("For C=", value, "the minDCF is", m)
                    print("")
                print("end c")
                plt = plot.plotDCFpoly(C, min_dcf, "C")
                fileName = ""
                bal = "UnBalanced"
                if(pi_T != 0):
                    bal = "Balanced"
                if(gauss): 
                    fileName = "plots/Poly/" + bal + "_" + "PolySVM" + o.strip() + ".pdf" #+ "_g=" + g + ".pdf"
                else: 
                    fileName = "plots/Poly/RAW" + bal + "_" + "PolySVM" + o.strip() + ".pdf" #+ "_g=" + g + ".pdf"
                plt.savefig(fileName)            

def NonLinearSVM(D,L):
    
    
    print("RBF")
    for gauss in [True, False]:
        if (gauss): print("gauss")
        else : print("raw")
        for b in [0, 0.5, 0.1, 0.9]:
            for p in priors:
                svm = SVM.SVM("RBF", pi = b,  C=10**(0.1), gamma = 10**(-0.3))
                min_dcf = generics.Kfold(D, L, svm, 0, gauss=gauss, prior=p)
                print("SVM RBF NO PCA pi_T=%.3f with C = 10^0.1 prior=0.5 ->  %.3f" %(b,min_dcf))

    print("Poly")
    for gauss in [True, False]:
        if (gauss): print("gauss")
        else : print("raw")
        for b in [0, 0.5, 0.1, 0.9]: #balanced or unbalanced
            if(b == 0):
                print("unbalanced")
            else:
                print("balanced: pi_T= " + str(b))
            for p in priors:
                svm = SVM.SVM("poly", pi=b, C=0.1, c = 1)
                min_dcf = generics.Kfold(D, L, svm, 0, gauss=gauss, prior=p)
                print("SVM Poly NO PCA Unbalanced with C = 0.1  prior=0.5 ->  %.3f" %(min_dcf))
    
                
def estimateComponentsInGMM(D, L):
    components = [2, 4, 8, 16, 32] 
    PCAoptions = {"NoPCA": 0}
    models = {"GMM", "GMMTied", "GMMDiag"}
    
    min_dcf = []
    for model in models:
        for gauss in [False, True]:
        #for gauss in [False]:
            if(gauss): print("Gauss Features")
            else: print("Raw features")
            for o in PCAoptions:
                min_dcf = []
                fileName = "plots/GMM/" + model + "_"
                if(gauss): 
                    fileName += "Gauss_"
                else: 
                    fileName += "Raw_"
                fileName += o
                fileName += ".pdf"
                print(fileName)
            
                for i in range(len(priors)):
                    tmp = []
                    for c in range(len(components)):
                        #m = GMM(components[c])
                        m = GMM(components[c], model)
                        #if(model == "GMMTied"):
                        #    m = GMMTiedCov(components[c])
                        #elif(model == "GMMDiag"):
                        #    m = GMMDiag(components[c])
                        res = generics.Kfold(D, L, m, PCAoptions[o], gauss=gauss, prior=priors[i])
                        tmp.append(res) 
                        print("For", components[c], "components and prior=", priors[i], "the minDCF of the GMM model is", tmp[c])
                    min_dcf.append(tmp)
                print("END")
                plt = plot.plotDCFGMM(components, min_dcf, "GMM components", fileName)

def compareDCFs(D, L):
    models = {#"MVG Full" : GaussianClassifier(), 
              "Quad LR" : LR.QuadLR(1e-2, 0.5), 
              "QuadSVM" : SVM.SVM("poly", pi=0, C=1, c = 1), 
              "RBF" :  SVM.SVM("RBF", pi=0,  C=10**(0.1), gamma=10**(-0.5)), 
              "GMM" : GMM(8,"GMMFull")
              }
    #models = {"GMM" : GMM(8,"GMMFull")}
    pis = numpy.linspace(-3, 3, 21)
    g = False
    for m in models:
        print("")
        print("model: ", m)
        if(m == "MVG Full"):
            g = True
        else:
            g = False
        for p in priors:
            minDCF, actDCF = generics.Kfold(D, L, models[m], 0, gauss=g, prior=p, ACT=True)        
            print("  for prior= ", p , " the min DCF is", minDCF, "and the actual DCF is", actDCF)
            

def calibrateModels(D,L):    

    models = { "Quad LR" : LR.QuadLR(1e-2, 0.5), 
              "QuadSVM" : SVM.SVM("poly", pi=0, C=1, c = 1), 
              "RBF" :  SVM.SVM("RBF", pi=0,  C=10**(0.1), gamma=10**(-0.5)), 
              "GMM" : GMM(8,"GMMFull")
    }
    g = False
    
    for m in models: 
        print(m)
        pis = numpy.linspace(-3, 3, 21)
        actualDCFs = []
        minDCFs = []
        for p in pis:
            pi = 1.0 / (1.0 + numpy.exp(-p))
            min_DCF, act_DCF = generics.Kfold(D, L, models[m], 0, gauss=g, prior=pi, ACT=True)
            actualDCFs.append(act_DCF)
            minDCFs.append(min_DCF)
        plot.bayesErrorPlot(actualDCFs, minDCFs, pis, m, "plots/Calibration/BayesError" + m + ".pdf")
        
        ##calibration
        print("calibration")
        act_dcf = []
        min_dcf = []
            
        for p in pis:
            
            pi = 1.0 / (1.0 + numpy.exp(-p))
            minDCFtmp, actDCFtmp = generics.Kfold(D, L, models[m], 0, gauss=g, prior=pi, calibrate=True)
            min_dcf.append(minDCFtmp)
            act_dcf.append(actDCFtmp)
        plot.bayesErrorPlot(act_dcf, min_dcf, pis, m, "plots/Calibration/Calibrated" + m + ".pdf")
        print("")
        
def evaluation(DTR,LTR, DTE, LTE):
    DTR, mu, sigma = generics.normalize_zscore(DTR)
    DTE, mu, sigma = generics.normalize_zscore(DTE, mu, sigma)
    
    #DTRPCA10, P = PCA.PCA(DTR, 10) 
    #DTEPCA10 = numpy.dot(P.T, DTE)
    
    NDTR, NDTE =  generics.gaussianization(DTR, DTE)

    #NDTRPCA10, P = PCA.PCA(NDTR, 10) 
    #NDTEPCA10 = numpy.dot(P.T, NDTE)
    
    
    models = {
              #"MVG Full" : GaussianClassifier(), 
              #"MVG Diag":  GaussianClassifierNaiveBayes(), 
              #"MVG Tied" : GaussianClassifierTiedCov(), 
              #"MVG TiedDiag" :  GaussianClassifierTiedDiagCov(),
              #"LR" : LR.LogisticRegression(1e-2, 0.5),
              #"SVM" :  SVM.SVM("linear", pi = 0, C=1e-1),
              "Quad LR" : LR.QuadLR(1e-2, 0.5), 
              "QuadSVM" : SVM.SVM("poly", pi=0, C=0.1, c = 1), 
              "RBF" :  SVM.SVM("RBF", pi=0,  C=10**(0.1), gamma=10**(-0.5)),
              "GMM" : GMM(8,"GMMFull")
              }
    for model in models:
        for g in [True, False]:
            if(g) : print("Gauss")
            else : print("raw")
            DR = []
            DE = []    
            for pi in priors:
                if g:
                    DR = NDTR
                    DE = NDTE
                else:
                    DR = DTR
                    DE = DTE
                models[model].train(DR, LTR)
                scores = models[model].predictAndGetScores(DE)
                min_DCF = minimum_detection_costs(scores, LTE, pi, 1, 1)
                
                print("    %s with prior=%.1f:  %.3f" %(model, pi, min_DCF))
            print("")
            
def evaluationHyperParam(DTR,LTR, DTE, LTE):
    DTR, mu, sigma = generics.normalize_zscore(DTR)
    DTE, mu, sigma = generics.normalize_zscore(DTE, mu, sigma)

    #RDTR, P = PCA.PCA(DTR, 7) #RAW DTR PCA = 7
    #RDTE = numpy.dot(P.T, DTE) #RAW DTE PCA = 7    
    NDTR, NDTE =  generics.gaussianization(DTR, DTE)
    print("eval hyperparam")
    
    print("LR")
    lambdas=numpy.logspace(-6, 2, num=25)
    min_dcf = []
    for pi in priors:
        #print("    Working on application with prior:", pi)
        for l in lambdas:
            lr = LR.LogisticRegression(l, 0.5)
            lr.train(DTR, LTR)
            scores = lr.predictAndGetScores(DTE)
            m = minimum_detection_costs(scores, LTE, pi, 1, 1)
            min_dcf.append(m)
            #print("      For lambda", l, "the minDCF is", m)
    plt = plot.plotDCF(lambdas, min_dcf, "λ")
    plt.savefig("plots/Evaluation/EvalRawLR"+ ".pdf")
    #print("\n")
    
    print("quadratic LR")
    lambdas=numpy.logspace(-6, 2, num=25)
    min_dcf = []
    for b in [0, 0.5]:
        min_dcf = []
        for pi in priors:
            #print("    Working on application with prior:", pi)
            for l in lambdas:
                lr = LR.QuadLR(l, 0.5)
                lr.train(DTR, LTR)
                scores = lr.predictAndGetScores(DTE)
                m = minimum_detection_costs(scores, LTE, pi, 1, 1)
                min_dcf.append(m)
                #print("      For lambda", l, "the minDCF is", m)
        plt = plot.plotDCF(lambdas, min_dcf, "λ")
        plt.savefig("plots/Evaluation/%sEvalRawQuadLR.pdf")
        #print("\n")
    
    print("SVM")
    C=numpy.logspace(-3, 1, num=15)
    
    for b in [0, 0.5]:
        min_dcf = []
        for pi in priors:
            #print("")
            #print("Working on application with prior:", pi)
            for value in C:
                svm = SVM.SVM("linear", pi=b, C=value)
                svm.train(DTR, LTR)
                scores = svm.predictAndGetScores(DTE)
                m = minimum_detection_costs(scores, LTE, pi, 1, 1)
                min_dcf.append(m)
                #print("  for prior=", pi , " the minDCF is", m)
            #print("")
        name = "plots/Evaluation/"
        if (b) : name += "balanced"
        else : name+= "unbalanced"
        name+="EvalLinearSVM.pdf"
        plt = plot.plotDCF(C, min_dcf, "C",)
        plt.savefig(name)
    
    
    print("Poly")
    min_dcf = []
    C = numpy.logspace(-3, 3, num=15)
    c = [0, 1, 10, 30]
    
    for b in [0, 0.5]:
        min_dcf = []
        for _c in c:
            #print("c: ", _c)
            for value in C:
                svm = SVM.SVM("poly", pi=b, C = value, c=_c)
                svm.train(DTR, LTR)
                scores = svm.predictAndGetScores(DTE)
                m = minimum_detection_costs(scores, LTE, 0.5, 1, 1)
                min_dcf.append(m)
            #print("")
        name = "plots/Evaluation/"
        if (b) : name += "balanced"
        else : name+= "unbalanced"
        name+="EvalRawPolySVM.pdf"
        plt = plot.plotDCFpoly(C, min_dcf, "C")
        plt.savefig(name)
    
    
    print("RBF")
    gamma = [10**(-1),10**(-0.7),10**(-0.5), 10**(-0.3), 10**(0.5)]
    C=numpy.logspace(-2, 2, num=15)

    for gauss in [True, False]:
        
        for b in [0, 0.5]:
            min_dcf = []
            #print("")
            #print("Working on application with prior: -- 0.5")
            for g in gamma:
                for value in C:
                    svm = SVM.SVM("RBF", pi=b, C=value, gamma=g)
                    scores = []
                    if(gauss):
                        svm.train(NDTR, LTR)
                        scores = svm.predictAndGetScores(NDTE)
                    else : 
                        svm.train(DTR, LTR)
                        scores = svm.predictAndGetScores(DTE)
                    m = minimum_detection_costs(scores, LTE, 0.5, 1, 1)
                    min_dcf.append(m)
            name = "plots/Evaluation/"
            if(gauss): name += "Gauss"
            else : name+= "Raw"
            if (b) : name += "balanced"
            else : name+= "unbalanced"
            name+="Eval_Raw_RBF_SVM.pdf"
            plt = plot.plotDCFRBF(C, min_dcf, "C")
            plt.savefig(name)

    min_dcf = []
    components = [2, 4, 8, 16, 32]  
    print("GMM")
    for pi in priors:
        tmp = []
        for c in range(len(components)):
            gmm = GMM(components[c], "GMM")
            gmm.train(DTR, LTR)
            scores = gmm.predictAndGetScores(DTE)
            m = minimum_detection_costs(scores, LTE, pi, 1, 1)
            tmp.append(m) 
            #print("For", components[c], "components and prior=", pi, "the minDCF of the GMM model is", tmp[c])
        min_dcf.append(tmp)
    plot.plotDCFGMM(components, min_dcf, "components","plots/Evaluation/Eval_RawGMM.pdf")     
    
    
def ROC(DTR, LTR, DTE, LTE):
    FPR = []
    TPR = []
    lambd = 1e-2
    split = 3
    prior = 0.5

    DTR, mu, sigma = generics.normalize_zscore(DTR)
    DTE, mu, sigma = generics.normalize_zscore(DTE, mu, sigma)

    #NDTR, NDTE =  generics.gaussianization(DTR, DTE)
    
    print("GMM")
    gmm = GMM(8, "GMM")
    gmm.train(DTR, LTR)
    scores = gmm.predictAndGetScores(DTE)
    scores, labels = generics.calibrateScores(scores, LTE, lambd, prior, evalu=True)

    sortedScores=numpy.sort(scores)

    for t in sortedScores:
        m = generics.computeOptBayesDecision(
            prior, 1, 1, scores, labels, t)
        FPRtemp, TPRtemp = generics.computeFPRTPR(prior, 1, 1, m)
        FPR.append(FPRtemp)
        TPR.append(TPRtemp)
    
    scores = []
    print("Quadratic Logistic Regression")
    FPR1 = []
    TPR1 = []
    lr = LR.QuadLR(1e-2, 0.5)
    lr.train(DTR, LTR)
    scores = lr.predictAndGetScores(DTE)
   
    scores, labels = generics.calibrateScores(scores, LTE, lambd, prior, evalu=True)    
    sortedScores=numpy.sort(scores)
    for t in sortedScores:
        m = generics.computeOptBayesDecision(
            prior, 1, 1, scores, labels, t)
        FPRtemp, TPRtemp = generics.computeFPRTPR(prior, 1, 1, m)
        FPR1.append(FPRtemp)
        TPR1.append(TPRtemp)
    
    print("Quadratic SVM")
    scores = []
    FPR2 = []
    TPR2 = []
    svm = SVM.SVM("poly", pi=0, C=0.1, c = 1)
    svm.train(DTR, LTR)
    scores = svm.predictAndGetScores(DTE)
    scores, labels = generics.calibrateScores(scores, LTE, lambd, prior, evalu=True)
    sortedScores=numpy.sort(scores)
    for t in sortedScores:
        m = generics.computeOptBayesDecision(
            prior, 1, 1, scores, labels, t)
        FPRtemp, TPRtemp = generics.computeFPRTPR(prior, 1, 1, m)
        FPR2.append(FPRtemp)
        TPR2.append(TPRtemp)

    print("RBF SVM")
    scores = []
    FPR3 = []
    TPR3 = []
    
    
    
    svm = SVM.SVM("RBF", pi=0, C=10**(0.1), gamma=10**(-0.5))
    svm.train(DTR, LTR)
    scores = svm.predictAndGetScores(DTE)
    scores, labels = generics.calibrateScores(scores, LTE, lambd, prior, evalu=True)
    
    sortedScores=numpy.sort(scores)
    for t in sortedScores:
        m = generics.computeOptBayesDecision(
            prior, 1, 1, scores, labels, t)
        FPRtemp, TPRtemp = generics.computeFPRTPR(prior, 1, 1, m)
        FPR3.append(FPRtemp)
        TPR3.append(TPRtemp)
   
    plot.plotROC(FPR, TPR, FPR1, TPR1, FPR2, TPR2, FPR3, TPR3)
    
 

if __name__ == "__main__":
    DTR, LTR = load("Train.txt")
    DTE, LTE = load("Test.txt")

    #featureAnalysis(DTR, LTR)
    
    #validation
    #MVG(DTR, LTR)
    #compareModelLR(DTR, LTR)
    #logisticRegression(DTR, LTR)
    #hyperparamsQuadLR(DTR, LTR)
    #QuadlogisticRegression(DTR, LTR)
    #compareModelSVM(DTR, LTR)
    #SVM_minDCF(DTR, LTR)
    #estimateRBFSVM(DTR, LTR)
    #estimatePolySVM(DTR, LTR)
    #NonLinearSVM(DTR, LTR)
    #estimateComponentsInGMM(DTR, LTR)
    #compareDCFs(DTR, LTR)
    #calibrateModels(DTR, LTR)
    
    #evaulation
    evaluation(DTR, LTR, DTE, LTE)
    #evaluationHyperParam(DTR, LTR, DTE, LTE)
    #ROC(DTR, LTR, DTE, LTE)