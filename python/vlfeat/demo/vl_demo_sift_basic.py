import Image
import numpy
import pylab
import vlfeat
from vlfeat.plotop.vl_plotframe import vl_plotframe
from numpy.random import shuffle


if __name__ == '__main__':
	# VL_DEMO_SIFT_BASIC  Demo: SIFT: basic functionality
	
	# --------------------------------------------------------------------
	#                     Load a figure and convert the to required format
	# --------------------------------------------------------------------
	I = Image.open('../../../data/a.jpg')
	I = vlfeat.vl_rgb2gray(numpy.array(I))	
	I = numpy.array(I, 'f', order='F') # 'F' = column-major order!

	pylab.gray()
	pylab.imshow(I)	
	print 'sift_basic_1'
	
	# --------------------------------------------------------------------
	#                                                             Run SIFT
	# --------------------------------------------------------------------
	f, d = vlfeat.vl_sift(I)
	sel = numpy.arange(f.shape[1])	
	shuffle(sel)
	vl_plotframe(f[:, sel[:50]], color='k', linewidth=3)
	vl_plotframe(f[:, sel[:50]], color='y')
	
	print 'sift_basic_2'
	
#	h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
#	set(h3,'color','k','linewidth',2) ;
#	h4 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
#	set(h4,'color','g','linewidth',1) ;
#	h1   = vl_plotframe(f(:,sel)) ; set(h1,'color','k','linewidth',3) ;
#	h2   = vl_plotframe(f(:,sel)) ; set(h2,'color','y','linewidth',2) ;
#	
#	vl_demo_print('sift_basic_3') ;

	# --------------------------------------------------------------------
	#                                                      Custom keypoint
	# --------------------------------------------------------------------
	pylab.figure()
	pylab.imshow(I[:200,:200]) 

	fc = numpy.array([99, 99, 10, -numpy.pi/8], 'float64')
	fc = numpy.array(numpy.atleast_2d(fc).transpose(), order='F')
	f, d = vlfeat.vl_sift(I, frames=fc, verbose=False)


#	h3   = vl_plotsiftdescriptor(d,f) ;  set(h3,'color','k','linewidth',3) ;
#	h4   = vl_plotsiftdescriptor(d,f) ;  set(h4,'color','g','linewidth',2) ;
	vl_plotframe(f, color='k', linewidth=4)
	vl_plotframe(f, color='y', linewidth=2)
	
	print 'sift_basic_4'

	# --------------------------------------------------------------------
	#                                   Custom keypoints with orientations
	# --------------------------------------------------------------------
	
	fc = numpy.array([99, 99, 10, 0], 'float64')
	fc = numpy.array(numpy.atleast_2d(fc).transpose(), order='F')
	f, d = vlfeat.vl_sift(I, frames=fc, orientations=True) ;
	
#	h3   = vl_plotsiftdescriptor(d,f) ;  set(h3,'color','k', 'linewidth',3) ;
#	h4   = vl_plotsiftdescriptor(d,f) ;  set(h4,'color','g', 'linewidth',2) ;
	vl_plotframe(f, color='k', linewidth=4)
	vl_plotframe(f, color='y', linewidth=2)
	
	print 'sift_basic_5'

	pylab.show()
	
	
	