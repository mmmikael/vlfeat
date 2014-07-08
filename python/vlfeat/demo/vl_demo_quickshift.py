import numpy
import Image
import vlfeat
#from vlfeat.plotop.vl_plotframe import vl_plotframe
import pylab

if __name__ == '__main__':
    """ VL_DEMO_QUICKSHIFT Demo: Quickshift segmentation
    """
    I = Image.open('../../../data/a.jpg')
    I=numpy.array(I)/255.
    pylab.figure()
    pylab.imshow(I)

    print 'quickshift_0'
    ratio = 0.5
    kernelsize = 2
    maxdist = 10
    seg=vlfeat.vl_quickseg(I,ratio,kernelsize,maxdist)
    pylab.figure()
    pylab.imshow(seg[0])

    print 'quickshift_1'
    kernelsize = 2
    maxdist = 20
    seg=vlfeat.vl_quickseg(I,ratio,kernelsize,maxdist)
    pylab.figure()
    pylab.imshow(seg[0])

    print 'quickshift_2'
    maxdist = 50
    ndists = 10
    seg=vlfeat.vl_quickvis(I,ratio,kernelsize,maxdist,ndists)
    pylab.figure()
    pylab.gray()
    pylab.imshow(seg[0])

    #pylab.show()
