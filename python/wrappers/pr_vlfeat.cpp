/*
 * pr_vlfeat.cpp
 *
 *  Created on: Apr 1, 2009
 *      Author: Mikael Rousson
 */


#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/overloads.hpp>

#define PY_ARRAY_UNIQUE_SYMBOL PyArrayVlfeat
#include <numpy/arrayobject.h> // in python/lib/site-packages/....

#include "vl_feat.h"


using namespace boost::python;
using namespace std;


void* extract_pyarray(PyObject* x)
{
	return x;
}

BOOST_PYTHON_MODULE(_vlfeat)
{
	converter::registry::insert(
	    &extract_pyarray, type_id<PyArrayObject>());

	def("vl_mser", vl_mser_python);
	def("vl_erfill", vl_erfill_python);
	def("vl_sift", vl_sift_python);

	import_array();
}

