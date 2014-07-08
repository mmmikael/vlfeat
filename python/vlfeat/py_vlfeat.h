/*
 * py_vlfeat.h
 *
 *  Created on: Apr 30, 2009
 *      Author: Mikael Rousson
 */

#pragma once

#include <Python.h>
#include <vector>

#define PY_ARRAY_UNIQUE_SYMBOL PyArrayVlfeat
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h> // in python/lib/site-packages/


#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/overloads.hpp>


/**
 * Computes maximally stable extremal regions
 * @param pyArray
 * @param delta MSER delta parameter
 * @param max_area Maximum region (relative) area ([0,1])
 * @param min_area Minimum region (relative) area ([0,1])
 * @param max_variation Maximum absolute region stability (non-negative)
 * @param min_diversity In-diversity argument must be in the [0,1] rang.
 * @return
 */
PyObject * vl_mser_python(
		PyArrayObject & pyArray,
		double delta = -1,
		double max_area = -1,
		double min_area = .05,
		double max_variation = -1,
		double min_diversity = -1);

/**
 *
 * @param image
 * @param seed
 * @return
 */
PyObject * vl_erfill_python(PyArrayObject & image, double seed);

/**
 *
 * @param image
 * @return
 */
PyObject * vl_sift_python(
		PyArrayObject & image,
		PyArrayObject & opt_frames,
		int opt_octaves = -1,
		int opt_levels = -1,
		int opt_first_octave = -1,
		double opt_peak_thresh = -1,
		double opt_edge_thresh = -1,
		double opt_norm_thresh = -1,
		double opt_magnif = -1,
		double opt_window_size = -1,
		bool opt_orientations = false,
		int opt_verbose = 0);

/**
 *
 * @param pyArray
 * @param opt_step
 * @param opt_bounds
 * @param opt_size
 * @param opt_fast
 * @param opt_verbose
 * @param opt_norm
 * @return
 */
PyObject * vl_dsift_python(
		PyArrayObject & pyArray,
		int opt_step,
		PyArrayObject & opt_bounds,
		int opt_size,
		bool opt_fast,
		bool opt_verbose,
		bool opt_norm);

/**
 *
 * @param in_grad
 * @param in_frames
 * @return
 */
PyObject * vl_siftdescriptor_python(
		PyArrayObject & in_grad,
		PyArrayObject & in_frames);

/**
 *
 * @param image
 * @param sigma
 * @return
 */
PyObject * vl_imsmooth_python(PyArrayObject & image, double sigma);

/**
 *
 * @param H
 * @param X
 * @param B
 * @param DIM
 * @return
 */
PyObject * vl_binsum_python(
		PyArrayObject & H,
		PyArrayObject & X,
		PyArrayObject & B,
		int DIM);



