import _vlfeat
import numpy

def vl_sift(
		data, 
		frames=numpy.zeros(1), 
		octaves=-1, 
		levels=-1, 
		first_octave=-1,
		peak_thresh=-1.0, 
		edge_thresh=-1.0,
		norm_thresh=-1.0,
		magnif=-1.0,
		window_size=-1.0,
		orientations=False,
		verbose=0):
	""" Computes the SIFT frames [1] (keypoints) F of the image I. I is a 
	gray-scale image in single precision. Each column of F is a feature frame 
	and has the format [X;Y;S;TH], where X,Y is the (fractional) center of the 
	frame, S is the scale and TH is the orientation (in radians). 
	Computes the SIFT descriptors [1] as well. Each column of D is the 
	descriptor of the corresponding frame in F. A descriptor is a 
	128-dimensional vector of class UINT8. 
	
	@param data         A gray-scale image in single precision 
	                    (float numpy array).
	@param frames       Set the frames to use (bypass the detector). If frames 
	                    are not passed in order of increasing scale, they are 
	                    re-orderded. 
	@param octaves      Set the number of octave of the DoG scale space. 
	@param levels       Set the number of levels per octave of the DoG scale 
	                    space. The default value is 3.
	@param first_octave Set the index of the first octave of the DoG scale 
	                    space. The default value is 0.
	@param peak_thresh  Set the peak selection threshold. 
	                    The default value is 0. 
	@param edge_thresh  Set the non-edge selection threshold. 
	                    The default value is 10.
	@param norm_thresh  Set the minimum l2-norm of the descriptor before 
	                    normalization. Descriptors below the threshold are set 
	                    to zero.
	@param magnif       Set the descriptor magnification factor. The scale of 
	                    the keypoint is multiplied by this factor to obtain the
	                    width (in pixels) of the spatial bins. For instance, if
	                    there are there are 4 spatial bins along each spatial
	                    direction, the ``diameter'' of the descriptor is
	                    approximatively 4 * MAGNIF. The default value is 3.
	@param orientations Compute the orientantions of the frames overriding the 
	                    orientation specified by the 'Frames' option.
	@param verbose      Be verbose (may be repeated to increase the verbosity
	                    level). 
	"""
	return _vlfeat.vl_sift(data, frames, octaves, levels, first_octave, 
						peak_thresh, edge_thresh, norm_thresh, magnif,
						window_size, orientations, verbose)

def vl_mser(
		data, 
		delta=5, 
		max_area=.75, 
		min_area=.0002,
		max_variation=.25, 
		min_diversity=.2):
	""" Computes the Maximally Stable Extremal Regions (MSER) [1] of image I 
	with stability threshold DELTA. I is any array of class UINT8. R is a vector
	of region seeds. \n 
	A (maximally stable) extremal region is just a connected component of one of
	the level sets of the image I. An extremal region can be recovered from a
	seed X as the connected component of the level set {Y: I(Y) <= I(X)} which
	contains the pixel o index X. \n
	It also returns ellipsoids F fitted to the regions. Each column of F 
	describes an ellipsoid; F(1:D,i) is the center of the elliposid and
	F(D:end,i) are the independent elements of the co-variance matrix of the
	ellipsoid. \n
	Ellipsoids are computed according to the same reference frame of I seen as 
	a matrix. This means that the first coordinate spans the first dimension of
	I. \n
	The function vl_plotframe() is used to plot the ellipses.
	
	@param data           A gray-scale image in single precision.
	@param delta          Set the DELTA parameter of the VL_MSER algorithm. 
	                      Roughly speaking, the stability of a region is the
	                      relative variation of the region area when the
	                      intensity is changed of +/- Delta/2. 
	@param max_area       Set the maximum area (volume) of the regions relative 
	                      to the image domain area (volume). 
	@param min_area       Set the minimum area (volume) of the regions relative 
	                      to the image domain area (volume). 
	@param max_variation  Set the maximum variation (absolute stability score) 
	                      of the regions. 
	@param min_diversity  Set the minimum diversity of the region. When the 
	                      relative area variation of two nested regions is below 
	                      this threshold, then only the most stable one is 
	                      selected. 
	"""
	return _vlfeat.vl_mser(data, delta, max_area, min_area, \
							max_variation, min_diversity)
	

def vl_erfill(data, r):
	""" Returns the list MEMBERS of the pixels which belongs to the extremal
	region represented by the pixel ER. \n
	The selected region is the one that contains pixel ER and of intensity 
	I(ER). \n
	I must be of class UINT8 and ER must be a (scalar) index of the region
	representative point. 
	"""
	return _vlfeat.vl_erfill(data, r)


def vl_dsift(
			pyArray, 
			step=-1, 
			bounds=numpy.zeros(1, 'f'), 
			size=-1, 
			fast=False, 
			verbose=False, 
			norm=False):
	return _vlfeat.vl_dsift(pyArray, step, bounds, size, fast, verbose, norm)


def vl_siftdescriptor(grad, frames):
	""" D = VL_SIFTDESCRIPTOR(GRAD, F) calculates the SIFT descriptors of the 
	keypoints F on the pre-processed image GRAD. GRAD is a 2xMxN array. The 
	first layer GRAD(1,:,:) contains the modulus of gradient of the original 
	image modulus. The second layer GRAD(2,:,:) contains the gradient angle 
	(measured in radians, clockwise, starting from the X axis -- this assumes 
	that the Y axis points down). The matrix F contains one column per keypoint 
	with the X, Y, SGIMA and ANLGE parameters. \n \n

	In order to match the standard SIFT descriptor, the gradient GRAD should be 
	calculated after mapping the image to the keypoint scale. This is obtained 
	by smoothing the image by a a Gaussian kernel of variance equal to the scale 
	of the keypoint. Additionaly, SIFT assumes that the input image is 
	pre-smoothed at scale 0.5 (this roughly compensates for the effect of the 
	CCD integrators), so the amount of smoothing that needs to be applied is 
	slightly less. The following code computes a standard SIFT descriptor by 
	using VL_SIFTDESCRIPTOR(): 
	"""
	return _vlfeat.vl_siftdescriptor(grad, frames)

def vl_imsmooth(I, sigma):
	""" I=VL_IMSMOOTH(I,SIGMA) convolves the image I by an isotropic Gaussian 
	kernel of standard deviation SIGMA. I must be an array of doubles. IF the 
	array is three dimensional, the third dimension is assumed to span different
	channels (e.g. R,G,B). In this case, each channel is convolved 
	independently.
	"""
	return _vlfeat.vl_imsmooth(I, sigma)
