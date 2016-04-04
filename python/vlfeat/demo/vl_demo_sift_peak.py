""" VL_DEMO_SIFT_PEAK  Demo: SIFT: peak threshold """

import numpy as np
import matplotlib.pyplot as plt
import vlfeat
from vlfeat.plotop.vl_plotframe import vl_plotframe
from numpy.random import uniform

if __name__ == '__main__':

    tmp = uniform(0, 1, (100, 500))
    I = np.zeros([100, 500])
    I.ravel()[plt.find(tmp.ravel() <= 0.005)] = 1

    I = (np.ones([100, 1]) * np.r_[0:1:500j]) * I
    I[:, 0] = 0
    I[:, -1] = 0
    I[0, :] = 0
    I[-1, :] = 0

    I = np.array(I, 'f', order='F')
    I = 2 * np.pi * 4 ** 2 * vlfeat.vl_imsmooth(I, 4)
    I *= 255

    print 'sift_peak_0'

    I = np.array(I, 'f', order='F')
    tpr = [0, 10, 20, 30]
    for tp in tpr:
        f, d = vlfeat.vl_sift(I, peak_thresh=tp, edge_thresh=10000)

        plt.figure()
        plt.gray()
        plt.imshow(I)
        vl_plotframe(f, color='k', linewidth=3)
        vl_plotframe(f, color='y', linewidth=2)

    plt.show()
