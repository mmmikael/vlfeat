import Image
import numpy
import pylab
import _vlfeat as vlfeat
from plotop import vl_plotframe

def mser_test(fileName):
	""" Simple mser test. Tests vlfeat.vl_mser and vlfeat.vl_erfill.
	"""
	# load image and convert to grayscale 
	ic = Image.open(fileName)
	image = ic.convert('L')
	data = numpy.array(image)
	
	# mser paramteres
	delta = 10
	max_area = .3
	min_area = .0002
	max_variation = .2
	min_diversity = .7
	
	# call mser on data
	[r1, f1] = vlfeat.vl_mser(data, delta, max_area, min_area, \
							max_variation, min_diversity)
	
	# call mser on 255 - data
	[r2, f2] = vlfeat.vl_mser(255 - data, delta, max_area, min_area, \
							max_variation, min_diversity)
	
	# get filled mser with er_fill
	M1 = numpy.zeros(data.shape[0] * data.shape[1])
	for r in r1:
		s = vlfeat.vl_erfill(data, r)
		M1[s] = M1[s] + 1
	M1 = M1.reshape(data.shape)
		
	M2 = numpy.zeros(data.shape[0] * data.shape[1])
	for r in r2:
		s = vlfeat.vl_erfill(255 - data, r)	
		M2[s] = M2[s] + 1
	M2 = M2.reshape(data.shape)
		
	# plot ellipses and region boundaries	
	pylab.subplot(1,2,1)		
	pylab.gray()
	pylab.imshow(data)
	vl_plotframe(f1, color='#ebbd0c')
	vl_plotframe(f2)	
	pylab.subplot(1,2,2)	
	pylab.gray()
	pylab.imshow(data)
	pylab.contour(M1, N=1, V=[1], colors='y')
	pylab.contour(M2, N=1, V=[1], colors='g')
	

def sift_test(fileName):
	""" Simple sift test. Tests vlfeat.vl_sift.
	"""
	# load image and convert to grayscale 
	ic = Image.open(fileName)
	image = ic.convert('L')
	data = numpy.array(image, 'f')
	
	[f, d] = vlfeat.vl_sift(data)
	print d.shape
	
	pylab.figure()
	pylab.gray()
	pylab.imshow(data)
	vl_plotframe(f.transpose(), color='#ebbd0c')
	

if __name__ == '__main__':
	fileName = '../../data/box.pgm'
	#fileName = '../../data/a.jpg'
	mser_test(fileName)
	sift_test(fileName)
	pylab.show()




