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

#include "py_vlfeat.h"
#include "kmeans/vl_ikmeans.h"
#include "kmeans/vl_hikmeans.h"


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
	def("vl_dsift", vl_dsift_python);
	def("vl_siftdescriptor", vl_siftdescriptor_python);
	def("vl_imsmooth", vl_imsmooth_python);

	/// HKM --------------------------------------------------------------------
	hikmeans_export();

	class_<vector<VlHIKMTree_python> >("VlHIKMTreeVec")
		.def(vector_indexing_suite<std::vector<VlHIKMTree_python> >())
		.def("push_back", &std::vector<VlHIKMTree_python>::push_back)
		.def("size", &std::vector<VlHIKMTree_python>::size)
		;

	def("vl_ikmeans",      vl_ikmeans_python);
	def("vl_ikmeanspush",  vl_ikmeanspush_python);
	def("vl_binsum",       vl_binsum_python);
	def("vl_hikmeans",     vl_hikmeans_python);
	def("vl_hikmeanspush", vl_hikmeanspush_python);
	/// ------------------------------------------------------------------------


	import_array();
}

