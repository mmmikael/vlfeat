""" VL_DEMO_SIFT_OR  """

import numpy as np
import matplotlib.pyplot as plt
# FIXME : Reorganize import statements.
import vlfeat
from vlfeat.plotop.vl_plotframe import vl_plotframe
from vlfeat.test.vl_test_pattern import vl_test_pattern

if __name__ == '__main__':
    # create pattern
    I = vl_test_pattern(1)

    # create frames (spatial sampling)
    ur = np.arange(0, I.shape[1], 5)
    vr = np.arange(0, I.shape[0], 5)
    [u, v] = np.meshgrid(ur, vr)

    f = np.array([u.ravel(), v.ravel()])
    K = f.shape[1]
    f = np.vstack([f, 2 * np.ones([1, K]), 0 * np.ones([1, K])])

    # convert frame to good format
    f = np.array(f, order='F')

    # compute sift
    f, d = vlfeat.vl_sift(np.array(I, 'f', order='F'),
                          frames=f, orientations=True)

    # display results
    plt.gray()
    plt.imshow(I)
    vl_plotframe(f, color='k', linewidth=3)
    vl_plotframe(f, color='y', linewidth=2)
    plt.show()

    print 'sift_or'
