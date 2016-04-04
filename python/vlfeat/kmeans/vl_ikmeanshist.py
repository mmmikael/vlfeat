import numpy as np
import vlfeat

def vl_ikmeanshist(K, asgn):

    """ VL_IKMEANSHIST  Compute histogram of quantized data
        H = VL_IKMEANSHIST(K,ASGN) computes the histogram of
        the IKM clusters activated by cluster assignments ASGN. """

    h = np.zeros(K, 'float64')
    h = vl_binsum(h, np.array([1], 'float64'), np.array(asgn, 'float64'))
    return h
