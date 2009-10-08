import numpy
import pylab
import vlfeat
from vlfeat.plotop.vl_plotframe import vl_plotframe

def plot_partition(data, datat, C, A, AT):
	K = C.shape[1]	
	colors = ['r', 'g', 'b', 'y', '#444444']
	for k in range(K):
		sel = pylab.find(A==k)
		selt = pylab.find(AT==k)
		vl_plotframe(data[:,sel],  color=colors[k])
		vl_plotframe(datat[:,selt], color=colors[k])

	pylab.plot(C[0,:],C[1,:],'ko', markersize=14, linewidth=6)
	pylab.plot(C[0,:],C[1,:],'yo', markersize=10, linewidth=1)
	

def vl_test_ikmeans():
	# VL_TEST_IKMEANS Test VL_IKMEANS function
	print 'test_ikmeans: Testing VL_IKMEANS and IKMEANSPUSH'

	# -----------------------------------------------------------------------
	print 'test_ikmeans: Testing Lloyd algorithm'
	
	K       = 5
	data    = numpy.array(numpy.random.rand(2,1000) * 255, 'uint8')
	datat   = numpy.array(numpy.random.rand(2,10000)* 255, 'uint8')
	
	[C, A] = vlfeat.vl_ikmeans(data, K, verbose=1)
	AT = vlfeat.vl_ikmeanspush(datat, C, verbose=1)
	plot_partition(data, datat, C, A, AT) 
	pylab.title('vl_ikmeans (Lloyd algorithm)')
	pylab.xlim(0, 255)
	pylab.ylim(0, 255)
	print 'ikmeans_lloyd'
	
	pylab.figure()
	[C, A] = vlfeat.vl_ikmeans(data, K, verbose=1, method='elkan')
	AT = vlfeat.vl_ikmeanspush(datat, C, verbose=1, method='elkan')
	pylab.title('vl_ikmeans (Elkan algorithm)')
	plot_partition(data, datat, C, A, AT) 
	pylab.xlim(0, 255)
	pylab.ylim(0, 255)
	print 'ikmeans_elkan'
	
	pylab.show()	

if __name__ == '__main__':
	vl_test_ikmeans()
