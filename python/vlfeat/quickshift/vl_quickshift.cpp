/*
 * vl_quickshift.cpp
 *
 * Author: Andreas Mueller
 *
 */

#include "vl_quickshift.h"

VlQS* vl_quickshift_new_python(PyArrayObject& image, int height, int width, int channels){
		vl_quickshift_new((vl_qs_type*)image.data,height,width,channels);	
}

PyObject * vl_quickshift_get_parents_python(VlQS const *q){
	npy_intp* dims=new npy_intp[2];
	dims[1]=q->width;
	dims[0]=q->height;
	PyArrayObject * out = (PyArrayObject*) PyArray_NewFromDescr( &PyArray_Type, PyArray_DescrFromType(PyArray_INT), 2, dims, NULL, NULL, NPY_F_CONTIGUOUS, NULL);
	memcpy(out->data,vl_quickshift_get_parents(q),sizeof(int)*dims[0]*dims[1]);
	delete dims;	
	return PyArray_Return(out);
	}

PyObject * vl_quickshift_get_dists_python(VlQS const *q){
	npy_intp* dims=new npy_intp[2];
	dims[1]=q->width;
	dims[0]=q->height;
	PyArrayObject * out = (PyArrayObject*) PyArray_NewFromDescr( &PyArray_Type, PyArray_DescrFromType(PyArray_DOUBLE), 2, dims, NULL, NULL, NPY_F_CONTIGUOUS, NULL);
	memcpy(out->data,vl_quickshift_get_dists(q),sizeof(double)*dims[0]*dims[1]);
	delete dims;	
	return PyArray_Return(out);
	}

PyObject * vl_quickshift_get_density_python(VlQS const *q){
	npy_intp* dims=new npy_intp[2];
	dims[1]=q->width;
	dims[0]=q->height;
	PyArrayObject * out = (PyArrayObject*) PyArray_NewFromDescr( &PyArray_Type, PyArray_DescrFromType(PyArray_DOUBLE), 2, dims, NULL, NULL, NPY_F_CONTIGUOUS, NULL);
	memcpy(out->data,vl_quickshift_get_density(q),sizeof(double)*dims[0]*dims[1]);
	delete dims;	
	return PyArray_Return(out);
	}
