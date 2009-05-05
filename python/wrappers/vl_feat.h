/*
 * vl_erfill_python.h
 *
 *  Created on: Apr 30, 2009
 *      Author: Mikael Rousson
 */

#pragma once

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL PyArrayVlfeat
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h> // in python/lib/site-packages/
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
PyObject * vl_sift_python(PyArrayObject & image);

