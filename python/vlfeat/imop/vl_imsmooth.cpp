/** @internal
 ** @file   imsmooth.c
 ** @author Andrea Vedaldi
 ** @brief  Smooth an image - MEX definition
 **/

/* AUTORIGHTS
 Copyright 2007 (c) Andrea Vedaldi and Brian Fulkerson

 This file is part of VLFeat, available in the terms of the GNU
 General Public License version 2.
 */

#include "../py_vlfeat.h"

extern "C" {
#include <vl/generic.h>
#include <vl/mathop.h>
#include <vl/imopv.h>
}

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <iostream>

/* option codes */
enum
{
	opt_padding = 0, opt_subsample, opt_kernel, opt_verbose
};


enum
{
	GAUSSIAN, TRIANGULAR
};

PyObject * vl_imsmooth_python(PyArrayObject & image, double sigma)
{
	assert(image.descr->type_num == PyArray_FLOAT);
	assert(image.flags & NPY_FORTRAN);

	int opt;

	int padding = VL_PAD_BY_CONTINUITY;
	int kernel = GAUSSIAN;
	int flags;
	int subsample = 1;
	int verb = 0;

	int M, N, K, j, k, ndims;
	int M_, N_;
	npy_intp dims_[3];
	int *dims;

	/* -----------------------------------------------------------------
	 *                                               Check the arguments
	 * -------------------------------------------------------------- */

//	if (nin < 2) {
//		mexErrMsgTxt("At least two input arguments required.");
//	} else if (nout > 1) {
//		mexErrMsgTxt("Too many output arguments.");
//	}

//	while ((opt = uNextOption(in, nin, options, &next, &optarg)) >= 0) {
//		switch (opt) {
//		case opt_padding: {
//			enum
//			{
//				buflen = 32
//			};
//			char buf[buflen];
//			if (!uIsString(optarg, -1)) {
//				mexErrMsgTxt("PADDING argument must be a string");
//			}
//			mxGetString(optarg, buf, buflen);
//			buf[buflen - 1] = 0;
//			if (uStrICmp("zero", buf) == 0) {
//				padding = VL_PAD_BY_ZERO;
//			} else if (uStrICmp("continuity", buf) == 0) {
//				padding = VL_PAD_BY_CONTINUITY;
//			} else {
//				mexErrMsgTxt("PADDING can be either ZERO or CONTINUITY");
//			}
//			break;
//		}
//
//		case opt_subsample:
//			if (!uIsRealScalar(optarg)) {
//				mexErrMsgTxt("SUBSAMPLE must be a scalar");
//			}
//			subsample = *mxGetPr(optarg);
//			if (subsample < 1) {
//				mexErrMsgTxt("SUBSAMPLE must be not less than one");
//			}
//			break;
//
//		case opt_kernel: {
//			enum
//			{
//				buflen = 32
//			};
//			char buf[buflen];
//			if (!uIsString(optarg, -1)) {
//				mexErrMsgTxt("KERNEL argument must be a string");
//			}
//			mxGetString(optarg, buf, buflen);
//			buf[buflen - 1] = 0;
//			if (uStrICmp("gaussian", buf) == 0) {
//				kernel = GAUSSIAN;
//			} else if (uStrICmp("triangular", buf) == 0) {
//				kernel = TRIANGULAR;
//			} else {
//				mexErrMsgTxt("Unknown kernel type");
//			}
//			break;
//		}
//
//		case opt_verbose:
//			++verb;
//			break;
//
//		default:
//			assert(0);
//		}
//	}

//	if (!uIsRealScalar(in[IN_S])) {
//		mexErrMsgTxt("S must be a real scalar");
//	}
//
//	classid = mxGetClassID(in[IN_I]);
//
//	if (classid != mxDOUBLE_CLASS && classid != mxSINGLE_CLASS) {
//		mexErrMsgTxt("I must be either DOUBLE or SINGLE.");
//	}
//
//	if (mxGetNumberOfDimensions(in[IN_I]) > 3) {
//		mexErrMsgTxt("I must be either a two or three dimensional array.");
//	}

	ndims = image.nd; //mxGetNumberOfDimensions(in[IN_I]);
	dims = new int[3];

	if(ndims>2) {
		dims[0] = image.dimensions[0];
		dims[1] = image.dimensions[1];
		dims[2] = image.dimensions[2]; //mxGetDimensions(in[IN_I]);
	} else {
		dims[0] = image.dimensions[0];
		dims[1] = image.dimensions[1]; //mxGetDimensions(in[IN_I]);
	}
	M = dims[0];
	N = dims[1];
	K = (ndims > 2) ? dims[2] : 1;

//	sigma = *mxGetPr(in[IN_S]);
//	if ((sigma < 0.01) && (subsample == 1)) {
//		out[OUT_J] = mxDuplicateArray(in[IN_I]);
//		return;
//	}

	M_ = (M - 1) / subsample + 1;
	N_ = (N - 1) / subsample + 1;
	dims_[0] = M_;
	dims_[1] = N_;
	if (ndims > 2)
		dims_[2] = ndims;

	PyArrayObject * out = (PyArrayObject*) PyArray_NewFromDescr(
		&PyArray_Type, PyArray_DescrFromType(PyArray_FLOAT),
		ndims, dims_, NULL, NULL, NPY_F_CONTIGUOUS, NULL);


	if (verb) {
		char const *classid_str = 0, *kernel_str = 0, *padding_str = 0;
		switch (padding) {
		case VL_PAD_BY_ZERO:
			padding_str = "with zeroes";
			break;
		case VL_PAD_BY_CONTINUITY:
			padding_str = "by continuity";
			break;
		default:
			assert(0);
			break;
		}
//		switch (classid) {
//		case mxDOUBLE_CLASS:
//			classid_str = "DOUBLE";
//			break;
//		case mxSINGLE_CLASS:
//			classid_str = "SINGLE";
//			break;
//		default:
//			assert(0);
//			break;
//		}
		switch (kernel) {
		case GAUSSIAN:
			kernel_str = "Gaussian";
			break;
		case TRIANGULAR:
			kernel_str = "triangular";
			break;
		default:
			assert(0);
			break;
		}

		printf(
			"imsmooth: [%dx%d] -> [%dx%d] (%s, sampling per. %d)\n", N, M, N_,
			M_, classid_str, subsample);
		printf("          padding: %s\n", padding_str);
		printf("          kernel: %s\n", kernel_str);
		printf("          sigma: %g\n", sigma);
		printf("          SIMD enabled: %s\n", vl_get_simd_enabled() ? "yes"
				: "no");
	}

	/* -----------------------------------------------------------------
	 *                                                        Do the job
	 * -------------------------------------------------------------- */
	flags = padding;
	flags |= VL_TRANSPOSE;

//	switch (classid) {
//	case mxSINGLE_CLASS:
#undef FLT
#undef VL_IMCONVCOL
#undef VL_IMCONVCOLTRI
#define FLT float
#define VL_IMCONVCOL vl_imconvcol_vf
#define VL_IMCONVCOLTRI vl_imconvcoltri_f
#include "imsmooth.tc"
//		break;
//	case mxDOUBLE_CLASS:
//#undef FLT
//#undef VL_IMCONVCOL
//#undef VL_IMCONVCOLTRI
//#define FLT double
//#define VL_IMCONVCOL vl_imconvcol_vd
//#define VL_IMCONVCOLTRI vl_imconvcoltri_vd
//#include "imsmooth.tc"
//		break;
//	default:
//		assert(0);
//		break;
//	}
	return PyArray_Return(out);
}
