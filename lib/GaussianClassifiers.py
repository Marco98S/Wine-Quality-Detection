import lib.generics as generics
import numpy 
import lib.multivariateGaussian as multivariateGaussian
import numpy.matlib 

class GaussianClassifier:
    
    def train (self, D, L):
        self.mean0 = generics.mcol(D[:, L == 0].mean(axis=1))
        self.mean1 = generics.mcol(D[:, L == 1].mean(axis=1))
    
        
        self.sigma0 = numpy.cov(D[:, L == 0])
        self.sigma1 = numpy.cov(D[:, L == 1])
        
        
        #class priors
        self.pi0 = D[:, L==0].shape[1]/D.shape[1]
        self.pi1 = D[:, L==1].shape[1]/D.shape[1]
         
     
    def predict (self, X):
        LS0 = multivariateGaussian.logpdf_GAU_ND(X, self.mean0, self.sigma0 )
        LS1 = multivariateGaussian.logpdf_GAU_ND(X, self.mean1, self.sigma1 )
        
        LS = numpy.vstack((LS0, LS1))
        
        #Log SJoints, that is the joint log-probabilities for a given sample
        LSJoint =  multivariateGaussian.joint_log_density(LS, generics.mcol(numpy.array([numpy.log(self.pi0), numpy.log(self.pi1) ])))
        
        #marginal log densities
        MLD = multivariateGaussian.marginal_log_densities(LSJoint)
        
        #Log-posteriors
        LP = multivariateGaussian.log_posteriors(LSJoint, MLD)
       
        predictions = numpy.argmax(LP, axis=0)
        
        return  predictions
    
    def predictAndGetScores (self, X):
        LS0 = multivariateGaussian.logpdf_GAU_ND(X, self.mean0, self.sigma0 )
        LS1 = multivariateGaussian.logpdf_GAU_ND(X, self.mean1, self.sigma1 )
        #log-likelihood ratios
        llr = LS1-LS0
        return llr

class GaussianClassifierNaiveBayes:
    
    def train (self, D, L):
        self.mean0 = generics.mcol(D[:, L == 0].mean(axis=1))
        self.mean1 = generics.mcol(D[:, L == 1].mean(axis=1))
    
        I=numpy.matlib.identity(D.shape[0])
        self.sigma0 =   numpy.multiply(numpy.cov(D[:, L == 0]),I)
        self.sigma1 =   numpy.multiply(numpy.cov(D[:, L == 1]),I)
    
        #class priors
        self.pi0 = D[:, L==0].shape[1]/D.shape[1]
        self.pi1 = D[:, L==1].shape[1]/D.shape[1]
     
    def predict (self, X):
        LS0 = numpy.asarray(multivariateGaussian.logpdf_GAU_ND(X, self.mean0, self.sigma0 )).flatten()
        LS1 = numpy.asarray(multivariateGaussian.logpdf_GAU_ND(X, self.mean1, self.sigma1 )).flatten()
      
        LS = numpy.vstack((LS0, LS1))
        
        #Log SJoints, that is the joint log-probabilities for a given sample
        LSJoint =  multivariateGaussian.joint_log_density(LS, generics.mcol(numpy.array([numpy.log(self.pi0), numpy.log(self.pi1) ])))
        
        #marginal log densities
        MLD = multivariateGaussian.marginal_log_densities(LSJoint)
        
        #Log-posteriors
        LP = multivariateGaussian.log_posteriors(LSJoint, MLD)
        return  numpy.argmax(LP, axis=0)
    
    def predictAndGetScores (self, X):
        LS0 = numpy.asarray(multivariateGaussian.logpdf_GAU_ND(X, self.mean0, self.sigma0 )).flatten()
        LS1 = numpy.asarray(multivariateGaussian.logpdf_GAU_ND(X, self.mean1, self.sigma1 )).flatten()
        #log-likelihood ratios
        llr = LS1-LS0
        return llr

class GaussianClassifierTiedCov:
    
    def train (self, D, L):
        self.mean0 = generics.mcol(D[:, L == 0].mean(axis=1))
        self.mean1 = generics.mcol(D[:, L == 1].mean(axis=1))
        
        self.sigma0 = numpy.cov(D[:, L == 0])
        self.sigma1 = numpy.cov(D[:, L == 1])

    
        self.sigma = 1/(D.shape[1])*(D[:, L == 0].shape[1]*self.sigma0+D[:, L == 1].shape[1]*self.sigma1)
    
        
        #class priors
        self.pi0 = D[:, L==0].shape[1]/D.shape[1]
        self.pi1 = D[:, L==1].shape[1]/D.shape[1]
      
    def predict (self, X):
        LS0 = multivariateGaussian.logpdf_GAU_ND(X, self.mean0, self.sigma )
        LS1 = multivariateGaussian.logpdf_GAU_ND(X, self.mean1, self.sigma )
       
      
        LS = numpy.vstack((LS0, LS1))
        
        #Log SJoints, that is the joint log-probabilities for a given sample
        LSJoint =  multivariateGaussian.joint_log_density(LS, generics.mcol(numpy.array([numpy.log(self.pi0), numpy.log(self.pi1) ])))
        
        #marginal log densities
        MLD = multivariateGaussian.marginal_log_densities(LSJoint)
        
        #Log-posteriors
        LP = multivariateGaussian.log_posteriors(LSJoint, MLD)
        
        
        return  numpy.argmax(LP, axis=0)
    
    def predictAndGetScores (self, X):
        LS0 = multivariateGaussian.logpdf_GAU_ND(X, self.mean0, self.sigma )
        LS1 = multivariateGaussian.logpdf_GAU_ND(X, self.mean1, self.sigma )
        #log-likelihood ratios
        llr = LS1-LS0
        return llr