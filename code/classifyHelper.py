import math
import numpy as np
import matplotlib.pyplot as plt
from imageHelper import imageHelper
from myMVND import MVND, log_likelihood
from typing import List
from imagePrior import get_prior





dataPath = '../data/'


def classify(img: imageHelper, mask: imageHelper, skin_mvnd: List[MVND], notSkin_mvnd: List[MVND], fig: str = "",
             prior_skin: float = 0.5, prior_nonskin: float = 0.5) -> None:
    '''
    :param img:             imageHelper object containing the image to be classified
    :param mask:            imageHelper object containing the ground truth mask
    :param skin_mvnd:            MVND object for the skin class
    :param notSkin_mvnd:            MVND object for the non-skin class
    :param fig:             Optional figure name
    :param prior_skin:      skin prior, float (0.0-1.0)
    :param prior_nonskin:   nonskin prior, float (0.0-1.0)
    '''
    im_rgb_lin = img.getLinearImage()
    if (type(skin_mvnd) != list):
        skin_mvnd = [skin_mvnd]
    if (type(notSkin_mvnd) != list):
        notSkin_mvnd = [notSkin_mvnd]
    log_likelihood_of_skin_rgb = log_likelihood(im_rgb_lin, skin_mvnd)
    log_likelihood_of_nonskin_rgb = log_likelihood(im_rgb_lin, notSkin_mvnd)

    testmask = mask.getLinearImageBinary().astype(int)[:, 0]
    npixels = len(testmask)

    log_likelihood_rgb = log_likelihood_of_skin_rgb - log_likelihood_of_nonskin_rgb
    #if log_likelihood_rgb >0, classify as skin, else as nonskin
    skin = (log_likelihood_rgb > 0).astype(int)
    print("mean of skin is: ", np.mean(skin))
    imgMinMask = skin - testmask
    print("imgminmask is: ", imgMinMask)
    print("imgMinMask shape is: ", imgMinMask.shape)

    # TODO: EXERCISE 2 - Error Rate without prior -> solved


    #double check
    fp = np.count_nonzero(imgMinMask == 1) / npixels
    fn = np.count_nonzero(imgMinMask == -1) / npixels

    totalError = (fp + fn) / npixels


    print('----- ----- -----')
    print('Total Error WITHOUT Prior =', totalError)
    print('false positive rate =', fp)
    print('false negative rate =', fn)

    #TODO: EXERCISE 2 - Error Rate with prior -> solved

    #calculate posterior probability
    log_likelihood_rgb = log_likelihood_of_skin_rgb * prior_skin - log_likelihood_of_nonskin_rgb * prior_nonskin

    skin_prior = (log_likelihood_rgb > 0).astype(int)

    imgMinMask_prior = skin_prior - testmask

    #double check
    fp_prior = np.count_nonzero(imgMinMask == 1) / npixels
    fn_prior = np.count_nonzero(imgMinMask == -1) / npixels

    totalError_prior = (fp + fn) / npixels


    print("posterior of skin rgb is: ", imgMinMask)

    print('----- ----- -----')
    print('Total Error WITH Prior =', totalError_prior)
    print('false positive rate =', fp_prior)
    print('false negative rate =', fn_prior)
    print('----- ----- -----')

    N = mask.N
    M = mask.M
    fpImage = np.reshape((imgMinMask > 0).astype(float), (N, M))
    fnImage = np.reshape((imgMinMask < 0).astype(float), (N, M))
    fpImagePrior = np.reshape((imgMinMask_prior > 0).astype(float), (N, M))
    fnImagePrior = np.reshape((imgMinMask_prior < 0).astype(float), (N, M))
    prediction = imageHelper()
    prediction.loadImage1dBinary(skin, N, M)
    predictionPrior = imageHelper()
    predictionPrior.loadImage1dBinary(skin_prior, N, M)


    plt.figure(fig)
    plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=2)
    plt.imshow(img.image)
    plt.axis('off')
    plt.title('Test image')

    plt.subplot2grid((4, 5), (0, 2), rowspan=2, colspan=2)
    plt.imshow(prediction.image, cmap='gray')
    plt.axis('off')
    plt.title('Skin prediction')

    plt.subplot2grid((4, 5), (2, 2), rowspan=2, colspan=2)
    plt.imshow(predictionPrior.image, cmap='gray')
    plt.axis('off')
    plt.title('Skin prediction PRIOR')

    plt.subplot2grid((4, 5), (2, 0), rowspan=2, colspan=2)
    plt.imshow(mask.image, cmap='gray')
    plt.axis('off')
    plt.title('GT mask')

    plt.subplot(4, 5, 5)
    plt.imshow(fpImage, cmap='gray')
    plt.axis('off')
    plt.title('FalsePositive')
    plt.subplot(4, 5, 10)
    plt.imshow(fnImage, cmap='gray')
    plt.axis('off')
    plt.title('FalseNegative')
    plt.subplot(4, 5, 15)
    plt.imshow(fpImagePrior, cmap='gray')
    plt.axis('off')
    plt.title('FalsePositive PRIOR')
    plt.subplot(4, 5, 20)
    plt.imshow(fnImagePrior, cmap='gray')
    plt.axis('off')
    plt.title('FalseNegative PRIOR')


