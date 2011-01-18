/** @internal
 ** @file     vl_dsift_python.cpp
 ** @author   Andrea Vedaldi
 ** @author   Mikael Rousson (Python wrapping)
 ** @brief    Dense Feature Transform (SIFT) - MEX
 **/

#include "../py_vlfeat.h"

/* AUTORIGHTS */

extern "C" {
#include <vl/mathop.h>
#include <vl/dsift.h>
}

#include <math.h>
#include <assert.h>

/** ------------------------------------------------------------------
 ** @brief Python entry point
 **/
PyObject * vl_dsift_python(
		PyArrayObject & pyArray,
		int opt_step,
		PyArrayObject & opt_bounds,
		int opt_size,
		bool opt_fast,
		bool opt_verbose,
		bool opt_norm)
{
	// check data type
	assert(pyArray.descr->type_num == PyArray_FLOAT);
	assert(pyArray.flags & NPY_FORTRAN);
	assert(opt_bounds.descr->type_num == PyArray_FLOAT);

	int verbose = 0;
	int opt;
	float const *data;
	int M, N;

	int step = 1;
	int size = 3;
	vl_bool norm = 0;

	vl_bool useFlatWindow = VL_FALSE;

	double *bounds = NULL;
	double boundBuffer[4];

	/* -----------------------------------------------------------------
	 *                                               Check the arguments
	 * -------------------------------------------------------------- */
	data = (float*) pyArray.data;
	M = pyArray.dimensions[0];
	N = pyArray.dimensions[1];

	if (opt_verbose)
		++verbose;
	if (opt_fast)
		useFlatWindow = 1;
	if (opt_norm)
		norm = 1;
	if (opt_bounds.nd == 1 && opt_bounds.dimensions[0] == 4) {
		double * tmp = (double *) opt_bounds.data;
		bounds = boundBuffer;
		for (int i = 0; i < 4; i++)
			bounds[i] = tmp[i];
	}
	if (opt_size >= 0)
		size = opt_size;
	if (opt_step >= 0)
		step = opt_step;


	// create PyTuple for outputs
	PyObject * tuple = PyTuple_New(2);

	/* -----------------------------------------------------------------
	 *                                                            Do job
	 * -------------------------------------------------------------- */
	{
		int numFrames;
		int descrSize;
		VlDsiftKeypoint const *frames;
		VlDsiftDescriptorGeometry const *geom;
		float const *descrs;
		int k, i;

		VlDsiftFilter *dsift;
		dsift = vl_dsift_new_basic(M, N, step, size);
		if (bounds) {
			vl_dsift_set_bounds(dsift, VL_MAX(bounds[0], 0), VL_MAX(
				bounds[1], 0), VL_MIN(bounds[2], M - 1), VL_MIN(bounds[3], N
					- 1));
		}
		vl_dsift_set_flat_window(dsift, useFlatWindow);

	    numFrames = vl_dsift_get_keypoint_num (dsift) ;
	    descrSize = vl_dsift_get_descriptor_size (dsift) ;
	    geom = vl_dsift_get_geometry(dsift);

		if (verbose) {
			int stepX;
			int stepY;
			int minX;
			int minY;
			int maxX;
			int maxY;
			vl_bool useFlatWindow;

			vl_dsift_get_steps(dsift, &stepX, &stepY);
			vl_dsift_get_bounds(dsift, &minX, &minY, &maxX, &maxY);
			useFlatWindow = vl_dsift_get_flat_window(dsift);

			printf("dsift: image size:        %d x %d\n", N, M);
			printf(
				"      bounds:            [%d, %d, %d, %d]\n", minY, minX,
				maxY, maxX);
			printf("      subsampling steps: %d, %d\n", stepY, stepX);
			printf(
				"      num bins:          [%d, %d, %d]\n", geom->numBinT,
				geom->numBinX, geom->numBinY);
			printf("      descriptor size:   %d\n", descrSize);
			printf(
				"      bin sizes:         [%d, %d]\n", geom->binSizeX,
				geom->binSizeY);
			printf("      flat window:       %s\n", VL_YESNO(useFlatWindow));
			printf("      number of frames:  %d\n", numFrames);
		}

		vl_dsift_process(dsift, data);

		frames = vl_dsift_get_keypoints(dsift);
		descrs = vl_dsift_get_descriptors(dsift);

		/* ---------------------------------------------------------------
		 *                                            Create output arrays
		 * ------------------------------------------------------------ */
		npy_intp dims[2];

		dims[0] = descrSize;
		dims[1] = numFrames;

		// allocate PyArray objects
		PyArrayObject * _descriptors = (PyArrayObject *) PyArray_NewFromDescr(
			&PyArray_Type, PyArray_DescrFromType(PyArray_UINT8),
			2, dims, NULL, NULL, NPY_F_CONTIGUOUS, NULL);

		if (norm)
			dims[0] = 3;
		else
			dims[0] = 2;

		PyArrayObject * _frames = (PyArrayObject*) PyArray_NewFromDescr(
			&PyArray_Type, PyArray_DescrFromType(PyArray_DOUBLE),
			2, dims, NULL, NULL, NPY_F_CONTIGUOUS, NULL);

		// put PyArray objects in PyTuple
		PyTuple_SetItem(tuple, 0, PyArray_Return(_frames));
		PyTuple_SetItem(tuple, 1, PyArray_Return(_descriptors));

		/* ---------------------------------------------------------------
		 *                                                       Copy back
		 * ------------------------------------------------------------ */
		{
			float *tmpDescr = (float*) vl_malloc(sizeof(float) * descrSize);

			double *outFrameIter = (double*) _frames->data;
			vl_uint8 *outDescrIter = (vl_uint8 *) _descriptors->data;
			for (k = 0; k < numFrames; ++k) {
				*outFrameIter++ = frames[k].y;
				*outFrameIter++ = frames[k].x;

				/* We have an implied / 2 in the norm, because of the clipping
				 below */
				if (norm)
					*outFrameIter++ = frames[k].norm;

				vl_dsift_transpose_descriptor(
					tmpDescr, descrs + descrSize * k, geom->numBinT,
					geom->numBinX, geom->numBinY);

				for (i = 0; i < descrSize; ++i) {
					*outDescrIter++ = (vl_uint8) (VL_MIN(
						512.0F * tmpDescr[i], 255.0F));
				}
			}
			vl_free(tmpDescr);
		}
		vl_dsift_delete(dsift);
	}

	return tuple;
}
