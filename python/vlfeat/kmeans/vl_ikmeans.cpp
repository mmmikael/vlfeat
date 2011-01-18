/* file:        ikmeans.c
 ** description: MEX weighted ikmeans function.
 ** author:      Andrea Vedaldi
 ** author:      Mikael Rousson (Python wrapping)
 **/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<assert.h>

extern "C" {
#include <vl/ikmeans.h>
#include <vl/generic.h>
}

#include "vl_ikmeans.h"

PyObject * vl_ikmeans_python(
		PyArrayObject & inputData,
		int K,
		int max_niters,
		char * method,
		int verb)
{
	// check types
	assert(inputData.descr->type_num == PyArray_UBYTE);
	assert(inputData.flags & NPY_FORTRAN);

	npy_intp M, N;
	int err = 0;

	vl_uint8 * data;

	VlIKMFilt *ikmf;

	M = inputData.dimensions[0]; /* n of components */
	N = inputData.dimensions[1]; /* n of elements */

	// K must be a positive integer not greater than the number of data.
	assert(K>0 && K<=N);

	int method_type = VL_IKM_LLOYD;

	if (strcmp("lloyd", method) == 0) {
		method_type = VL_IKM_LLOYD;
	} else if (strcmp("elkan", method) == 0) {
		method_type = VL_IKM_ELKAN;
	} else {
		assert(0);
	}

	/* ------------------------------------------------------------------
	 *                                                         Do the job
	 * --------------------------------------------------------------- */
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
		printf("ikmeans: MaxInters = %d\n", max_niters);
		printf("ikmeans: Method    = %s\n", method_name);
	}

	data = (vl_uint8*) inputData.data;
	ikmf = vl_ikm_new(method_type);

	vl_ikm_set_verbosity(ikmf, verb);
	vl_ikm_set_max_niters(ikmf, max_niters);
	vl_ikm_init_rand_data(ikmf, data, M, N, K);

	err = vl_ikm_train(ikmf, data, N);
	if (err)
		printf("ikmeans: possible overflow!");

	/* ------------------------------------------------------------------
	 *                                                       Return results
	 * --------------------------------------------------------------- */

	// allocate PyArrayObject for centers (column-majored)
	npy_intp dims[2];
	dims[0] = M;
	dims[1] = K;
	PyArrayObject * centers = (PyArrayObject*) PyArray_NewFromDescr(
		&PyArray_Type, PyArray_DescrFromType(PyArray_INT32),
		2, dims, NULL, NULL, NPY_F_CONTIGUOUS, NULL);

	// copy data
	int * centers_data = (int*) centers->data;
	memcpy(centers_data, vl_ikm_get_centers(ikmf), sizeof(vl_ikm_acc) * M * K);


	// allocate PyArrayObject for cluster assignment array
	PyArrayObject * asgn = (PyArrayObject*) PyArray_SimpleNew(
		1, (npy_intp*) &N, PyArray_UINT32);

	// copy data
	unsigned int * asgn_data = (unsigned int*) asgn->data;
	int j;
	vl_ikm_push(ikmf, (vl_uint*) asgn_data, data, N);

	// clean
	vl_ikm_delete(ikmf);

	if (verb) {
		printf("ikmeans: done\n");
	}

	// construct tuple to return both results: (regions, frames)
	PyObject * tuple = PyTuple_New(2);
	PyTuple_SetItem(tuple, 0, PyArray_Return(centers));
	PyTuple_SetItem(tuple, 1, PyArray_Return(asgn));

	return tuple;
}
