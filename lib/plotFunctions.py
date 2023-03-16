import matplotlib.pyplot as plt
import seaborn
import numpy

def plot_hist(D, L):

    print(D)
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
        0 : "Mean of the integrated profile",
        1 : "Standard deviation of the integrated profile",
        2 : "Excess kurtosis of the integrated profile",
        3 : "Skewness of the integrated profile",
        4 : "Mean of the DM-SNR curve",
        5 : "Standard deviation of the DM-SNR curve",
        6 : "Excess kurtosis of the DM-SNR curve",
        7 : "Skewness of the DM-SNR curve",
    }

    for dIdx in range(8):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'negative')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'positive')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        #plt.savefig('_hist_%d.pdf'  % dIdx)
    plt.show()


def plot_heatmap(D):
    plt.imshow(D)#, cmap='hot', interpolation='nearest')
    plt.show()
 
def plot_scatter(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    for dIdx1 in range(12):
        for dIdx2 in range(12):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(dIdx1)
            plt.ylabel(dIdx2)
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'male')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'female')
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            #plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()




def heatmap(D, L):
    plt.figure()
    seaborn.heatmap(numpy.corrcoef(D), linewidth=0.2, cmap="Greys", square=True, cbar=False)
    plt.figure()
    seaborn.heatmap(numpy.corrcoef(D[:, L==0]), linewidth=0.2, cmap="Reds", square=True,cbar=False)
    plt.figure()
    seaborn.heatmap(numpy.corrcoef(D[:, L==1]), linewidth=0.2, cmap="Blues", square=True, cbar=False)
    return 

def plotDCFGMM(x, y, xlabel, name):
    plt.figure()
    print(y)
    print(x)
    #plt.plot(numpy.power(x,2), y[0], label='min DCF prior=0.5', color='r')
    #plt.plot(numpy.power(x,2), y[1], label='min DCF prior=0.1', color='b')
    #plt.plot(numpy.power(x,2), y[2], label='min DCF prior=0.9', color='g')
    plt.plot(x, y[0], label='min DCF prior=0.5', color='r')
    plt.plot(x, y[1], label='min DCF prior=0.1', color='b')
    plt.plot(x, y[2], label='min DCF prior=0.9', color='g')
    plt.xlim([min(x), max(x)])
    #plt.xlim([2**min(x), 2**max(x)])
    plt.xscale("log", base=2)
    plt.legend(["min DCF prior=0.5", "min DCF prior=0.1", "min DCF prior=0.9"])

    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.savefig(name)
    return plt

def plotDCF(x, y, xlabel):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5', color='r')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.1', color='b')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.9', color='g')
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5", "min DCF prior=0.1", "min DCF prior=0.9"])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    return plt

def plotDCFpoly(x, y, xlabel):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5 - c=0', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.5 - c=1', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.5 - c=10', color='g')
    plt.plot(x, y[3*len(x): 4*len(x)], label='min DCF prior=0.5 - c=30', color='m')

    
    plt.xlim([1e-2, 1e2])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5 - c=0", "min DCF prior=0.5 - c=1", 
                'min DCF prior=0.5 - c=10', 'min DCF prior=0.5 - c=30'])

    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    return plt

def plotDCFRBF(x, y, xlabel):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5 - logγ=-1', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.5 - logγ=-0.7', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.5 - logγ=-0.5', color='g')
    plt.plot(x, y[3*len(x): 4*len(x)], label='min DCF prior=0.5 - logγ=-0.4', color='y')
    plt.plot(x, y[4*len(x): 5*len(x)], label='min DCF prior=0.5 - logγ=0.5', color='c')
    
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5 - logγ=-1", "min DCF prior=0.5 - logγ=-0.7", 
                'min DCF prior=0.5 - logγ=-0.5', 'min DCF prior=0.5 - logγ=-0.3',
                'min DCF prior=0.5 - logγ=0.5'])
    
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    return plt
    
def bayesErrorPlot(dcf, mindcf, effPriorLogOdds, model, name):
    plt.figure()
    plt.plot(effPriorLogOdds, dcf, label='act DCF', color='r')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b', linestyle="--")
    plt.xlim([min(effPriorLogOdds), max(effPriorLogOdds)])
    plt.legend([model + " - act DCF", model+" - min DCF"])
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF")
    plt.savefig(name)
    return

def bayesErrorPlot2DCF(dcf0, dcf1, mindcf, effPriorLogOdds, model, lambda0, lambda1, name):
    plt.figure()
    plt.plot(effPriorLogOdds, dcf0, label='act DCF', color='r')
    plt.plot(effPriorLogOdds, dcf1, label='act DCF', color='g')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b', linestyle="--")
    plt.xlim([min(effPriorLogOdds), max(effPriorLogOdds)])
    plt.legend([model + " - act DCF lambda = "+lambda0, model + " - act DCF lambda = "+lambda1, model+" - min DCF"])
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF")
    plt.savefig(name)
    return

def bayesErrorPlotSVM2DCF(dcf0, dcf1, mindcf, effPriorLogOdds, model, C0, C1, name):
    plt.figure()
    plt.plot(effPriorLogOdds, dcf0, label='act DCF', color='r')
    plt.plot(effPriorLogOdds, dcf1, label='act DCF', color='g')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b', linestyle="--")
    plt.xlim([min(effPriorLogOdds), max(effPriorLogOdds)])
    plt.legend([model + " - act DCF C = "+C0, model + " - act DCF C = "+C1, model+" - min DCF"])
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF")
    plt.savefig(name)
    return


def plotROC(FPR, TPR, FPR1, TPR1, FPR2, TPR2, FPR3, TPR3):
    # Function used to plot TPR(FPR)
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(FPR, TPR, linewidth=2, color='r')
    plt.plot(FPR1, TPR1, linewidth=2, color='b')
    plt.plot(FPR2, TPR2, linewidth=2, color='g')
    plt.plot(FPR3, TPR3, linewidth=2, color='y')
    plt.legend(["Gauss Full", "Quadratic LR", "Quadratic SVM", "RBF SVM"])
    
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig("ROC.pdf")
    return