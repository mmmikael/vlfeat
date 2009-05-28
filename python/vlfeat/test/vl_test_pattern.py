import numpy
import pylab
from numpy.random import uniform

def vl_test_pattern(n):
	""" VL_TEST_PATTERN  Generate test pattern
	I=VL_TEST_PATTERN(N) returns the N-th test pattern.
	"""
	
	ur = numpy.r_[-1:1:128j]
	vr = numpy.r_[-1:1:128j]
	[u, v] = numpy.meshgrid(ur, vr);
	
	if n == 1:
		#I	 = u.^2 + v.^2 > (1/4).^2
		I = numpy.abs(u) + numpy.abs(v) > (1.0 / 4)
		I = 255 * I
		I[:64, :] = 0 
	if n == 2:
		I = numpy.zeros([100, 100])
		I[20:100201, 20:100201] = 128
		I[30:100301, 30:100301] = 200
		I[50, 50]				 = 255
		I[50, 55]				 = 250
		I[50, 45]				 = 245
		I = 255 - I
#	if n == 3:
#		I = 255 * vl_imsmooth(checkerboard(10, 10), 1)
	if n == 4:
		I = 255 * uniform(0,1, (32, 32))
#	if n == 101:
#		I = 255 * vl_imreadbw(fullfile(vlfeat_root, 'data', 'a.jpg'))
#	if n == 102:
#		 I = 255 * vl_imreadbw(fullfile(vlfeat_root, 'data', 'box.pgm'))   
	if n == 'cone':
		I = numpy.sqrt(u**2 + v**2)
		
	return I

if __name__ == '__main__':
	I = vl_test_pattern(4)
	pylab.gray()
	pylab.imshow(I, interpolation='nearest')
	pylab.show()