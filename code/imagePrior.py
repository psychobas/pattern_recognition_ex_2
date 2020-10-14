import numpy as np
from imageHelper import imageHelper


def get_prior(mask: imageHelper) -> (float, float):
    [N, M] = mask.shape #shape is (400, 320)
    image_mask = mask.image[:]
    print("unique values are: ", np.unique(image_mask))
    print("image_mask is: ", image_mask.mean())
    # TODO: EXERCISE 2 - Compute the skin and nonskin prior
    #as the mask has either values 1 (for skin) or 0 (nonskin),
    # I use the mean to get the fraction of skin pixels (which is the prior for the skin
    prior_skin = image_mask.mean()
    prior_nonskin = 1 - image_mask.mean()
    print("prior_skin is: ", prior_skin)

    return prior_skin, prior_nonskin



