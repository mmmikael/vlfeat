import numpy

def vl_ertr(f):
	""" VL_ERTR  Transpose exremal regions frames 
	F = VL_ERTR(F) transposes the frames F as returned by VL_MSER(). This
	conversion is required as the VL_MSER algorithm considers the column index I
	as the frist image index, while according standard image convention the
	first coordinate is the abscissa X.
	"""
	assert(f.shape[0] == 5)

	res = numpy.zeros(f.shape)
	res[0, :] = f[1, :]
	res[1, :] = f[0, :]
	res[2, :] = f[4, :]
	res[3, :] = f[3, :]
	res[4, :] = f[2, :]
	
	return res
