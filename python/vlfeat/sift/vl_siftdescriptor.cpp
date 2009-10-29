/** @internal
 ** @file     vl_siftdescriptor_python.cpp
 ** @author   Andrea Vedaldi
 ** @author   Mikael Rousson (Python wrapping)
 ** @brief    SIFT descriptor - MEX
 **/

#include "../py_vlfeat.h"

extern "C" {
#include <vl/mathop.h>
#include <vl/sift.h>
}

#include <math.h>
#include <assert.h>

#include <iostream>

/* option codes */
enum
{
	opt_magnif, opt_verbose
};


/** ------------------------------------------------------------------
 ** @internal
 ** @brief Transpose descriptor
 **
 ** @param dst destination buffer.
 ** @param src source buffer.
 **
 ** The function writes to @a dst the transpose of the SIFT descriptor
 ** @a src. The transpose is defined as the descriptor that one
 ** obtains from computing the normal descriptor on the transposed
 ** image.
 **/

VL_INLINE void transpose_descriptor(vl_sift_pix* dst, vl_sift_pix* src)
{
	int const BO = 8; /* number of orientation bins */
	int const BP = 4; /* number of spatial bins     */
	int i, j, t;

	for (j = 0; j < BP; ++j) {
		int jp = BP - 1 - j;
		for (i = 0; i < BP; ++i) {
			int o = BO * i + BP * BO * j;
			int op = BO * i + BP * BO * jp;
			dst[op] = src[o];
			for (t = 1; t < BO; ++t)
				dst[BO - t + op] = src[t + o];
		}
	}
}

/** ------------------------------------------------------------------
 ** @brief MEX entry point
 **/

PyObject * vl_siftdescriptor_python(
		PyArrayObject & in_grad,
		PyArrayObject & in_frames)
{
	// TODO: check types and dim
	//	"GRAD must be a 2xMxN matrix of class SINGLE."

	assert(in_grad.descr->type_num == PyArray_FLOAT);
	assert(in_frames.descr->type_num == PyArray_FLOAT64);
	assert(in_grad.flags & NPY_FORTRAN);
	assert(in_frames.flags & NPY_FORTRAN);

	int verbose = 0;
	int opt;

	//  TODO: check if we need to do a copy of the grad array
	float * grad_array;
	vl_sift_pix *grad;
	int M, N;

	double magnif = -1;
	double *ikeys = 0;
	int nikeys = 0;

	int i, j;


	/* -----------------------------------------------------------------
	 *                                               Check the arguments
	 * -------------------------------------------------------------- */

	// get frames nb and data pointer
	nikeys = in_frames.dimensions[1];
	ikeys = (double *) in_frames.data;


	// TODO: deal with optional params
	//  while ((opt = uNextOption(in, nin, options, &next, &optarg)) >= 0) {
	//    switch (opt) {
	//
	//      case opt_verbose :
	//        ++ verbose ;
	//        break ;
	//
	//      case opt_magnif :
	//        if (!uIsRealScalar(optarg) || (magnif = *mxGetPr(optarg)) < 0) {
	//          mexErrMsgTxt("MAGNIF must be a non-negative scalar.") ;
	//        }
	//        break ;
	//
	//      default :
	//        assert(0) ;
	//        break ;
	//    }
	//  }

	//  TODO: convert to Python
	//grad_array = mxDuplicateArray(in[IN_GRAD]); (copy?)
	grad = (float*) in_grad.data; //mxGetData(grad_array);
	M = in_grad.dimensions[1];
	N = in_grad.dimensions[2];


	/* transpose angles */
	for (i = 1; i < 2 * M * N; i += 2) {
		grad[i] = VL_PI / 2 - grad[i];
	}

	/* -----------------------------------------------------------------
	 *                                                            Do job
	 * -------------------------------------------------------------- */
	PyArrayObject * _descr;
	{
		VlSiftFilt *filt = 0;
		vl_uint8 *descr = 0;

		/* create a filter to process the image */
		filt = vl_sift_new(M, N, -1, -1, 0);

		if (magnif >= 0)
			vl_sift_set_magnif(filt, magnif);

		if (verbose) {
			printf("siftdescriptor: filter settings:\n");
			printf(
				"siftdescriptor:   magnif                = %g\n",
				vl_sift_get_magnif(filt));
			printf("siftdescriptor:   num of frames         = %d\n", nikeys);
		}

		{
			npy_intp dims[2];
			dims[0] = 128;
			dims[1] = nikeys;

			_descr = (PyArrayObject*) PyArray_NewFromDescr(
				&PyArray_Type, PyArray_DescrFromType(PyArray_UBYTE),
				2, dims, NULL, NULL, NPY_F_CONTIGUOUS, NULL);

			descr = (vl_uint8*) _descr->data;
		}

		/* ...............................................................
		 *                                             Process each octave
		 * ............................................................ */
		for (i = 0; i < nikeys; ++i) {
			vl_sift_pix buf[128], rbuf[128];

			double y = *ikeys++;
			double x = *ikeys++;
			double s = *ikeys++;
			double th = VL_PI / 2 - *ikeys++;


			vl_sift_calc_raw_descriptor(filt, grad, buf, M, N, x, y, s, th);

			transpose_descriptor(rbuf, buf);

			for (j = 0; j < 128; ++j) {
				double x = 512.0 * rbuf[j];
				x = (x < 255.0) ? x : 255.0;
				*descr++ = (vl_uint8) (x);
			}
		}
		/* cleanup */
//		mxDestroyArray(grad_array);
		vl_sift_delete(filt);
	} /* job done */

	return PyArray_Return(_descr);
}
