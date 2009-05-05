/**
 ** @author   Andrea Vedaldi
 ** @author   Mikael Rousson (Python wrapping)
 ** @brief    MSER MEX driver - Python wrapper
 **/

#include "vl_feat.h"

#include <src/generic-driver.h>
#include <vl/generic.h>
#include <vl/stringop.h>
#include <vl/pgm.h>
#include <vl/mser.h>

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <iostream>


PyObject * vl_mser_python(
		PyArrayObject & pyArray,
		double delta,
		double max_area,
		double min_area,
		double max_variation,
		double min_diversity)
{

	unsigned char * data = (unsigned char *) pyArray.data;
	unsigned int width = pyArray.dimensions[1];
	unsigned int height = pyArray.dimensions[0];

	VlMserFilt *filt = 0;
	int i, j, dof, nregions, nframes, q;
	vl_uint const *regions;
	float const *frames;

	// image dims
	int ndims = 2;
	int dims[ndims];
	dims[0] = width;
	dims[1] = height;

	// mser filter
	filt = vl_mser_new(ndims, dims);

	// set parameters
	if (delta >= 0)
		vl_mser_set_delta(filt, (vl_mser_pix) delta);
	if (max_area >= 0)
		vl_mser_set_max_area(filt, max_area);
	if (min_area >= 0)
		vl_mser_set_min_area(filt, min_area);
	if (max_variation >= 0)
		vl_mser_set_max_variation(filt, max_variation);
	if (min_diversity >= 0)
		vl_mser_set_min_diversity(filt, min_diversity);

	// do mser computation
	vl_mser_process(filt, (vl_mser_pix*) data); // vl_mser_pix = unsigned char

	// fit ellipses
	vl_mser_ell_fit(filt);

	// get results
	nregions = vl_mser_get_regions_num(filt);
	regions = vl_mser_get_regions(filt);

	nframes = vl_mser_get_ell_num(filt);
	dof = vl_mser_get_ell_dof(filt);
	frames = vl_mser_get_ell(filt);

	// convert results to PyArrayObjects
	int odims[2];
	odims[0] = dof;
	odims[1] = nframes;

	// allocate pyarray objects
	PyArrayObject * _regions = (PyArrayObject*) PyArray_SimpleNew(
		1, &nregions, PyArray_DOUBLE);

	PyArrayObject * _frames = (PyArrayObject*) PyArray_SimpleNew(
		2, odims, PyArray_FLOAT);

	// check if valid pointers
	assert(_regions);
	assert(_frames);

	// fill pyarray objects
	double * _regions_buf = (double *) _regions->data;
	for (i = 0; i < nregions; ++i) {
		_regions_buf[i] = regions[i];
	}

	float * _frames_buf = (float *) _frames->data;
	int k = 0;
	for (j = 0; j < dof; ++j) {
		for (i = 0; i < nframes; ++i) {
			_frames_buf[k++] = frames[i * dof + j]; //+ ((j < ndims) ? 1.0 : 0.0);
		}
	}

    /* cleanup */
    vl_mser_delete (filt) ;

	// construct tuple to return both results: (regions, frames)
	PyObject * tuple = PyTuple_New(2);
	PyTuple_SetItem(tuple, 0, PyArray_Return(_regions));
	PyTuple_SetItem(tuple, 1, PyArray_Return(_frames));

	return tuple;
}





