import Image
import numpy
import pylab
import vlfeat
from vlfeat.plotop.vl_plotframe import vl_plotframe
from numpy.random import uniform

if __name__ == '__main__':
	""" VL_DEMO_SIFT_PEAK  Demo: SIFT: peak treshold
	"""
	tmp = uniform(0, 1, (100, 500)) 
	I = numpy.zeros([100, 500])
	I.ravel()[pylab.find(tmp.ravel()<=0.005)] = 1
	
	I = (numpy.ones([100,1]) *  numpy.r_[0:1:500j]) * I 
	I[:, 0] = 0
	I[:, -1] = 0
	I[0, :] = 0
	I[-1, :] = 0
	
	I = numpy.array(I, 'f', order='F')
	I = 2 * numpy.pi * 4**2 * vlfeat.vl_imsmooth(I, 4)	
	I = 255 * I

	print 'sift_peak_0'

	I = numpy.array(I, 'f', order='F')
	tpr = [0, 10, 20, 30]
	for tp in tpr:
		f, d = vlfeat.vl_sift(I, peak_thresh=tp, edge_thresh=10000)
		
		pylab.figure()
		pylab.gray()
		pylab.imshow(I)
		vl_plotframe(f, color='k', linewidth=3)
		vl_plotframe(f, color='y', linewidth=2)
	
	pylab.show()