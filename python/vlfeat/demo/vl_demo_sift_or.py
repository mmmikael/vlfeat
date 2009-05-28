import Image
import numpy
import pylab
import vlfeat
from vlfeat.plotop.vl_plotframe import vl_plotframe
from vlfeat.test.vl_test_pattern import vl_test_pattern

if __name__ == '__main__':
	# create pattern
	I = vl_test_pattern(1) ;
	
	# create frames (spatial sampling)
	ur = numpy.arange(0, I.shape[1], 5)
	vr = numpy.arange(0, I.shape[0], 5)	
	[u,v] = numpy.meshgrid(ur, vr) 
	
	f = numpy.array([u.ravel(), v.ravel()])
	K = f.shape[1]
	f = numpy.vstack([f, 2 * numpy.ones([1, K]), 0 * numpy.ones([1, K])])
	
	# convert frame to good format
	f = numpy.array(f, order='F')
	
	# compute sift
	f, d = vlfeat.vl_sift(numpy.array(I, 'f', order='F'), frames=f, orientations=True)
	
	# display results
	pylab.gray()
	pylab.imshow(I)
	vl_plotframe(f, color='k', linewidth=3)
	vl_plotframe(f, color='y', linewidth=2)
	pylab.show()
	
	print 'sift_or'
	
