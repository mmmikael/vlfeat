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
#include "quickshift/vl_quickshift.h"

extern "C" {
#include <vl/quickshift.h>
}

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

	// export quickshift low level stuff
	class_<VlQS> ("VlQS");
	def("vl_quickshift_new", vl_quickshift_new_python,return_value_policy<manage_new_object>());
	def("vl_quickshift_process", vl_quickshift_process);
	def("vl_quickshift_delete", vl_quickshift_delete);
	def("vl_quickshift_set_kernel_size", vl_quickshift_set_kernel_size);
	def("vl_quickshift_set_medoid", vl_quickshift_set_medoid);
	def("vl_quickshift_set_max_dist", vl_quickshift_set_max_dist);

	def("vl_quickshift_get_parents", vl_quickshift_get_parents_python);
	def("vl_quickshift_get_dists", vl_quickshift_get_dists_python);
	def("vl_quickshift_get_density", vl_quickshift_get_density_python);
	/// ------------------------------------------------------------------------

	import_array();
}

