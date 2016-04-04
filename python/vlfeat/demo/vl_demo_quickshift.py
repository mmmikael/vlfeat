""" VL_DEMO_QUICKSHIFT Demo: Quickshift segmentation """

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import vlfeat
# from vlfeat.plotop.vl_plotframe import vl_plotframe

if __name__ == '__main__':
    I = Image.open('../../../data/a.jpg')
    I = np.array(I)/255.
    plt.figure()
    plt.imshow(I)

    print 'quickshift_0'
    ratio = 0.5
    kernelsize = 2
    maxdist = 10
    seg = vlfeat.vl_quickseg(I, ratio, kernelsize, maxdist)
    plt.figure()
    plt.imshow(seg[0])

    print 'quickshift_1'
    kernelsize = 2
    maxdist = 20
    seg = vlfeat.vl_quickseg(I, ratio, kernelsize, maxdist)
    plt.figure()
    plt.imshow(seg[0])

    print 'quickshift_2'
    maxdist = 50
    ndists = 10
    seg = vlfeat.vl_quickvis(I, ratio, kernelsize, maxdist, ndists)
    plt.figure()
    plt.gray()
    plt.imshow(seg[0])

    plt.show()
