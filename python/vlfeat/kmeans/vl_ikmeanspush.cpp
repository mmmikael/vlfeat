/* file:        ikmeanspush.c
 ** description: MEX weighted ikmeanspush function.
 ** author:      Andrea Vedaldi
 ** author:      Mikael Rousson (Python wrapping)
 **/

/* AUTORIGHTS
 Copyright 2007 (c) Andrea Vedaldi and Brian Fulkerson

 This file is part of VLFeat, available in the terms of the GNU
 General Public License version 2.
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<assert.h>

extern "C" {
#include <vl/generic.h>
#include <vl/ikmeans.h>
}

#include "vl_ikmeans.h"

PyObject * vl_ikmeanspush_python(
		PyArrayObject & inputData,
		PyArrayObject & centers,
		char * method,
		int verb)
{

	// check types
	assert(inputData.descr->type_num == PyArray_UBYTE);
	assert(inputData.flags & NPY_FORTRAN);
	assert(centers.descr->type_num == PyArray_INT32);
	assert(centers.flags & NPY_FORTRAN);

	vl_ikm_acc* centers_data = (vl_ikm_acc *) centers.data;
	vl_uint8 * data = (vl_uint8 *) inputData.data;

	int j;
	npy_intp M, N, K = 0;

	VlIKMFilt *ikmf;

	/** -----------------------------------------------------------------
	 **                                               Check the arguments
	 ** -------------------------------------------------------------- */

	M = inputData.dimensions[0]; //mxGetM(in[IN_X]) ;  /* n of components */
	N = inputData.dimensions[1]; //mxGetN(in[IN_X]) ;  /* n of elements */
	K = centers.dimensions[1]; //mxGetN(in[IN_C]) ;  /* n of centers */

	if (centers.dimensions[0] != M) {
		printf("DATA and CENTERS must have the same number of columns.");
	}

	int method_type = VL_IKM_LLOYD;

	if (strcmp("lloyd", method) == 0) {
		method_type = VL_IKM_LLOYD;
	} else if (strcmp("elkan", method) == 0) {
		method_type = VL_IKM_ELKAN;
	} else {
		assert(0);
	}

	/** -----------------------------------------------------------------
	 **                                               Check the arguments
	 ** -------------------------------------------------------------- */

	if (verb) {
		char const * method_name = 0;
		switch (method_type) {
		case VL_IKM_LLOYD:
			method_name = "Lloyd";
			break;
		case VL_IKM_ELKAN:
			method_name = "Elkan";
			break;
		default:
			assert (0);
		}
		printf("ikmeanspush: Method = %s\n", method_name);
		printf("ikmeanspush: ndata  = %d\n", N);
	}

	PyArrayObject * asgn = (PyArrayObject *) PyArray_SimpleNew(
			1, (npy_intp*) &N, PyArray_UINT32);
	unsigned int * asgn_data = (unsigned int*) asgn->data;

	ikmf = vl_ikm_new(method_type);

	vl_ikm_set_verbosity(ikmf, verb);
	vl_ikm_init(ikmf, centers_data, M, K);
	vl_ikm_push(ikmf, asgn_data, data, N);

	vl_ikm_delete(ikmf);

	return PyArray_Return(asgn);
}
