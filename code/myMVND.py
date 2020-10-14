import numpy as np
from typing import List
from scipy.stats import multivariate_normal



#imports for debugging, delete later!!
import sys, os
import scipy.io
from imageHelper import imageHelper
from imagePrior import get_prior



class MVND:
    # TODO: EXERCISE 2 - Implement mean and covariance matrix of given data -> done
    #DONE
    def __init__(self, data: np.ndarray, c: float = 1.0):
        #dim of data is (3, 10000)
        self.c = c  # Mixing coefficients. The sum of all mixing coefficients = 1.0.
        self.data = data
        print(data.shape)
        self.mean = np.mean(self.data, axis = 1)
        print("mean is: ", self.mean)
        self.cov  = np.cov(self.data)
        print("cov is: ", self.cov)

        #debug
        #sample from mvn and check if results match
        #diagonal covariance
        mult_n = np.random.multivariate_normal([1,1], [[1,0], [0,5]])

    # TODO: EXERCISE 2 - Implement pdf and logpdf of a MVND -> done
    #pdf done, very similar results as scipy
    #scipy.stats.multivariate_normal.pdf(x = np.array([1,1]), mean = np.array([0,0]), cov = [[1,0], [0,1]])
    def pdf(self, x: np.ndarray) -> np.ndarray:       # Alternatively a float can also be returned if individual datapoints are computed

        """pdf of the multivariate normal distribution."""
        assert x.shape == self.mean.shape
        print(x.shape)
        x_centered = x - self.mean
        print("x_centered is: ", x_centered)

        d = self.cov.ndim
        print("d is: ", d)
        print(self.cov.shape)
        print(x_centered.shape)

        pdf = (1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov))) * np.exp(-(np.linalg.solve(self.cov, x_centered).T.dot(x_centered)) / 2))



        return pdf




#data will e.g., be im_rgb_lin
def log_likelihood(data: np.ndarray, mvnd: List[
    MVND]) -> np.ndarray:  # Alternatively a float can also be returned if individual datapoints are computed
    '''
    Compute the log likelihood of each datapoint
    :param data:    Training inputs, #(samples) x #(dim)
    :param mvnd:     List of MVND objects
    :return:        Likelihood of each data point
    '''
    #data.shape[0] is (128000, 3) for im_rgb_lin (which is passed to the data argument in classifyHelper
    log_likelihood = np.zeros((1, data.shape[0]))
    #log_likelihood will have shape (1, 12800)

    # TODO: EXERCISE 2 - Compute likelihood of data -> done
    # Note: For MVGD there will only be 1 item in the list
    for g in mvnd:
        #shape of cov is (3,3); ndarray
        #shape of mean is (3, ); ndarray
        x_centered = data - g.mean
        print("x_centered is: ", x_centered)
        print("shape of x_centered is: ", x_centered.shape)
        #determinant
        det_cov = np.linalg.det(g.cov)



        #see https://stackoverflow.com/questions/42178497/numpy-loglikelihood-of-multivariate-normal-distribution
        log_lik = -0.5*np.einsum('...j,jk,...k', x_centered, np.linalg.inv(g.cov), x_centered) -(x_centered.shape[1]/2) * np.log(2*np.pi) -(1/2)* np.log(det_cov)
        print(log_lik)
        print("log_lik shape is: ", log_lik.shape)

        # log_lik = -(x_centered.shape[0]*x_centered.shape[1]/2) * np.log(2 * np.pi) - x_centered.shape[0]/2 * np.log(det_cov)
        # print("log_lik is: ", log_lik)

        # log_lik = - x_centered.T.dot(np.linalg.inv(g.cov)).dot(x_centered) #-(x_centered.shape[1]/2) * np.log(2*np.pi) -(1/2)* np.log(det_cov)
        # print(log_lik) #has error in dimension

        #
        # loglikelihood = -0.5 * (
        #         np.log(np.linalg.det(cov))
        #         + np.einsum('...j,jk,...k', residuals, np.linalg.inv(cov), residuals)
        #         + len(mean) * np.log(2 * np.pi)
        # )
        # np.sum(loglikelihood)
        #
        # -0.5 * (np.log(np.linalg.det(cov)) + residuals.T.dot(np.linalg.inv(cov)).dot(residuals) + 2 * np.log(2 * np.pi))



        likelihood = log_lik
        return(likelihood)






        #log_likelihood_of_skin_rgb = log_likelihood(im_rgb_lin, skin_mvnd)
        #log_likelihood_of_nonskin_rgb = log_likelihood(im_rgb_lin, notSkin_mvnd)





    return log_likelihood



#check MVND function
mvnd_test_data = np.random.multivariate_normal([1,1], [[1,0], [0,1]], size = (50)).T
print("my test data shape is: ", mvnd_test_data.shape)
my_MVND = MVND(data = mvnd_test_data)
print("my mean is: ", my_MVND.mean)
print("my cov is: ", my_MVND.cov)





#debug
dataPath = '../data/'

#data for mvnd
sdata = scipy.io.loadmat(os.path.join(dataPath, 'skin.mat'))['sdata']
ndata = scipy.io.loadmat(os.path.join(dataPath, 'nonskin.mat'))['ndata']

mvn_sskin = [MVND(sdata)]
mvn_nskin = [MVND(ndata)]

# Optain priors
mask = imageHelper()
mask.loadImageFromFile(os.path.join(dataPath, 'mask.png'))
# TODO: EXERCISE 2 - Compute the skin and nonskin prior
prior_skin, prior_nonskin = get_prior(mask)


#load training and test image
trainingmaskObj = imageHelper()
trainingmaskObj.loadImageFromFile(os.path.join(dataPath, 'mask.png'))
trainingimageObj = imageHelper()
trainingimageObj.loadImageFromFile(os.path.join(dataPath, 'image.png'))

#input for log likelihood (img in classify, data in log_likelihood)
im_rgb_lin = trainingimageObj.getLinearImage()
#together with normal distribution of s/ndata (mvn_sskin, mvn_nskin



#test function
log_likelihood(im_rgb_lin, mvn_sskin)


