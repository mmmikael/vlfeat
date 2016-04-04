# import numpy as num
# FIXME : Unused import numpy. Check later.
import pylab
# TODO : "import matplotlib.pyplot as plt" Make sure this new import statement isn't causing issues.

eps = 1e-9


def vl_plotframe(frames, color='#00ff00', linewidth=2):
    # VL_PLOTFRAME  Plot feature frame
    # VL_PLOTFRAME(FRAME) plots the frames FRAME.  Frames are attributed
    # image regions (as, for example, extracted by a feature detector). A
    # frame is a vector of D=2,3,..,6 real numbers, depending on its
    # class. VL_PLOTFRAME() supports the following classes:
    #
    # * POINTS
    # + FRAME(1:2) coordinates
    #
    # * CIRCLES
    # + FRAME(1:2)   center
    # + FRAME(3)   radius
    #
    #   * ORIENTED CIRCLES
    # + FRAME(1:2)   center
    # + FRAME(3)    radius
    # + FRAME(4)   orientation
    #
    #   * ELLIPSES
    # + FRAME(1:2)   center
    # + FRAME(3:5)   S11, S12, S22 of x' inv(S) x = 1.
    #
    #   * ORIENTED ELLIPSES
    # + FRAME(1:2)   center
    # + FRAME(3:6)   A(:) of ELLIPSE = A UNIT_CIRCLE
    #
    #  H=VL_PLOTFRAME(...) returns the handle of the graphical object
    #  representing the frames.
    #
    #  VL_PLOTFRAME(FRAMES) where FRAMES is a matrix whose column are FRAME
    #  vectors plots all frames simultaneously. Using this call is much
    #  faster than calling VL_PLOTFRAME() for each frame.
    #
    #  VL_PLOTFRAME(FRAMES,...) passes any extra argument to the underlying
    #  plot function. The first optional argument can be a line
    #  specification string such as the one used by PLOT().
    #
    #  See also:: VL_HELP().

    # AUTORIGHTS
    # Copyright 2007 (c) Andrea Vedaldi and Brian Fulkerson
    #
    # This file is part of VLFeat, available in the terms of the GNU
    # General Public License version 2.

    # number of vertices drawn for each frame
    np = 40

    # --------------------------------------------------------------------
    # Handle various frame classes
    # --------------------------------------------------------------------

    # if just a vector, make sure it is column
    if pylab.min(frames.shape) == 1:
        frames = frames[:]

    [D, K] = frames.shape
    zero_dimensional = D == 2

    # just points?
    if zero_dimensional:
        h = pylab.plot(frames[0, :], frames[1, :], '.', color=color)
        return

    # reduce all other cases to ellipses/oriented ellipses
    frames = frame2oell(frames)
    do_arrows = (D == 4 or D == 6)

    # --------------------------------------------------------------------
    # Draw
    # --------------------------------------------------------------------

    K = frames.shape[1]
    thr = pylab.linspace(0, 2 * pylab.pi, np)

    # allx and ally are nan separated lists of the vertices describing the
    # boundary of the frames
    allx = pylab.nan * pylab.ones([np * K + (K - 1), 1])
    ally = pylab.nan * pylab.ones([np * K + (K - 1), 1])

    if do_arrows:
        # allxf and allyf are nan separated lists of the vertices of the
        allxf = pylab.nan * pylab.ones([3 * K])
        allyf = pylab.nan * pylab.ones([3 * K])

    # vertices around a unit circle
    Xp = pylab.pylab.array([pylab.cos(thr), pylab.sin(thr)])

    for k in range(K):
        # frame center
        xc = frames[0, k]
        yc = frames[1, k]

        # frame matrix
        A = frames[2:6, k].reshape([2, 2])

        # vertices along the boundary
        X = pylab.dot(A, Xp)
        X[0, :] = X[0, :] + xc
        X[1, :] = X[1, :] + yc

        # store
        allx[k * (np + 1) + pylab.arange(0, np), 0] = X[0, :]
        ally[k * (np + 1) + pylab.arange(0, np), 0] = X[1, :]

        if do_arrows:
            allxf[k * 3 + pylab.arange(0, 2)] = xc + [0, A[0, 0]]
            allyf[k * 3 + pylab.arange(0, 2)] = yc + [0, A[1, 0]]
    if do_arrows:
        for k in range(K):
            h = pylab.plot(allx[k * (np + 1) + pylab.arange(0, np), 0],
                           ally[k * (np + 1) + pylab.arange(0, np), 0],
                           color=color, linewidth=linewidth)
            h = pylab.plot(allxf[k * 3 + pylab.arange(0, 2)],
                           allyf[k * 3 + pylab.arange(0, 2)],
                           color=color, linewidth=linewidth)
    else:
        for k in range(K):
            h = pylab.plot(allx[k * (np + 1) + pylab.arange(0, np), 0],
                           ally[k * (np + 1) + pylab.arange(0, np), 0],
                           color=color, linewidth=linewidth)


def frame2oell(frames):
    """FRAMES2OELL  Convert generic frame to oriented ellipse
       EFRAMES = FRAME2OELL(FRAMES) converts the frames FRAMES to
       oriented ellipses EFRAMES. This is useful because many tasks are
       almost equivalent for all kind of regions and are immediately
       reduced to the most general case. """

    #
    # Determine the kind of frames
    #
    [D, K] = frames.shape

    if D == 2:
        kind = 'point'
    elif D == 3:
        kind = 'disk'
    elif D == 4:
        kind = 'odisk'
    elif D == 5:
        kind = 'ellipse'
    elif D == 6:
        kind = 'oellipse'
    else:
        print 'FRAMES format is unknown'
        raise

    eframes = pylab.zeros([6, K])

    #
    # Do conversion
    #
    if kind == 'point':
        eframes[0:2, :] = frames[0:2, :]
    elif kind == 'disk':
        eframes[0:2, :] = frames[0:2, :]
        eframes[2, :] = frames[2, :]
        eframes[5, :] = frames[4, :]
    elif kind == 'odisk':
        r = frames[2, :]
        c = r * pylab.cos(frames[3, :])
        s = r * pylab.sin(frames[3, :])

        eframes[1, :] = frames[1, :]
        eframes[0, :] = frames[0, :]
        # eframes[2:6, :] = [c, s, - s, c]
        eframes[2:6, :] = [c, -s, s, c]  # not sure why
    elif kind == 'ellipse':
        # sz = find(1e6 * abs(eframes(3,:)) < abs(eframes(4,:)+eframes(5,:))
        eframes[0:2, :] = frames[0:2, :]
        eframes[2, :] = pylab.sqrt(frames[2, :])
        eframes[3, :] = frames[3, :] / (eframes[2, :] + eps)
        eframes[4, :] = pylab.zeros([1, K])
        eframes[5, :] = pylab.sqrt(frames[4, :] -
                             frames[3, :] * frames[3, :] / (frames[2, :] + eps))
    elif kind == 'oellipse':
        eframes = frames

    return eframes
