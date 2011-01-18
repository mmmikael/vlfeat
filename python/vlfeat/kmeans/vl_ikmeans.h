/*
 * vl_hikmeans.h
 *
 *  Created on: Jul 13, 2009
 *      Author: Mikael Rousson
 */

#pragma once

#include "../py_vlfeat.h"

/**
 *
 * @param inputData
 * @param K
 * @param max_niters
 * @param methode
 * @param verb
 * @return
 */
PyObject * vl_ikmeans_python(
		PyArrayObject & inputData,
		int K,
		int max_niters = 200,
		char * methode = "lloyd",
		int verb = 0);

/**
 *
 * @param inputData
 * @param centers
 * @param method
 * @param verb
 * @return
 */
PyObject * vl_ikmeanspush_python(
		PyArrayObject & inputData,
		PyArrayObject & centers,
		char * method,
		int verb);

