/** file:        hikmeans.c
 ** description: MEX hierarchical ikmeans.
 ** author:      Brian Fulkerson
 ** author:      Mikael Rousson (Python wrapping)
 **
 **/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<assert.h>

extern "C" {
#include <vl/hikmeans.h>
#include <vl/generic.h>
}

#include "vl_hikmeans.h"


using namespace std;
using namespace boost::python;


// global object definition: we need that to create boost instances of our
// wrapped VlHIKMTree
object VlHIKMTree_python_;

void hikmeans_export()
{
	VlHIKMTree_python_ =
		class_<VlHIKMTree_python> ("VlHIKMTree", init<int, int> ())
			.def_readwrite("K", &VlHIKMTree_python::K)
			.def_readwrite("depth", &VlHIKMTree_python::depth)
			.def_readwrite("sub", &VlHIKMTree_python::sub)
			.def("get_centers", &VlHIKMTree_python::getCenters)
			.def("save", &VlHIKMTree_python::save)
			.def("load", &VlHIKMTree_python::load)
		;
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Copy HIKM tree node to a C++/Python structure
 **/
static void xcreate(VlHIKMTree_python & pnode, VlHIKMNode *node)
{
	// get nb nodes and dims
	int node_K = vl_ikm_get_K(node->filter);
	int M = vl_ikm_get_ndims(node->filter);

	// set centers
	vl_ikm_acc const *centers = vl_ikm_get_centers(node->filter);
	PyArrayObject * pcenters;
	npy_intp dims[2] = {M, node_K};
	pcenters = (PyArrayObject*) PyArray_NewFromDescr(
		&PyArray_Type, PyArray_DescrFromType(PyArray_INT32),
		2, dims, NULL, NULL, NPY_F_CONTIGUOUS, NULL);

	memcpy((int*)pcenters->data, centers, sizeof(vl_ikm_acc) * M * node_K);
	pnode.centers = (PyObject *) pcenters;

	// recursive call on children
	if (node->children) {
		for (int k = 0; k < node_K; ++k) {
			pnode.sub.push_back(VlHIKMTree_python(node_K, 0));
			VlHIKMTree_python & psub = pnode.sub.back();
			xcreate(psub, node->children[k]);
		}
	}
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Copy HIKM tree to a C++/Python structure
 **/
object
hikm_to_python(VlHIKMTree * tree)
{
	int K = vl_hikm_get_K(tree);
	int depth = vl_hikm_get_depth(tree);
	int dims[2] = { 1, 1 };

	object ptree_ = VlHIKMTree_python_(K, depth);
	VlHIKMTree_python & ptree = extract<VlHIKMTree_python&>(ptree_);

	if (tree->root)
		xcreate(ptree, tree->root);

	return ptree_;
}

/** ------------------------------------------------------------------
 ** @internal
 **/
tuple vl_hikmeans_python(
		PyArrayObject & data,
		int K,
		int nleaves,
		int verb,
		int max_iters,
		char * method)
{
	assert(data.descr->type_num == PyArray_UBYTE);
	assert(data.flags & NPY_FORTRAN);
	assert(nleaves>1);

	vl_uint8 * data_ptr;
	int M, N, depth = 0;

	int opt;
	int method_type = VL_IKM_LLOYD;

	VlHIKMTree* tree;

	/* ------------------------------------------------------------------
	 *                                                Check the arguments
	 * --------------------------------------------------------------- */

	M = data.dimensions[0]; /* n of components */
	N = data.dimensions[1]; /* n of elements */

	assert(K<N);

	data_ptr = (vl_uint8 *) data.data;

	if (strcmp("lloyd", method) == 0) {
		method_type = VL_IKM_LLOYD;
	} else if (strcmp("elkan", method) == 0) {
		method_type = VL_IKM_ELKAN;
	} else {
		printf("Unknown cost type.");
	}

	/* ---------------------------------------------------------------
	 *                                                      Do the job
	 * ------------------------------------------------------------ */

	depth = VL_MAX(1, ceil(log(nleaves) / log(K)));
	tree = vl_hikm_new(method_type);

	if (verb) {
		printf("hikmeans: # dims: %d\n", M);
		printf("hikmeans: # data: %d\n", N);
		printf("hikmeans: K: %d\n", K);
		printf("hikmeans: depth: %d\n", depth);
	}

	vl_hikm_set_verbosity(tree, verb);
	vl_hikm_init(tree, M, K, depth);
	vl_hikm_train(tree, data_ptr, N);

	npy_intp dims[2] = { vl_hikm_get_depth(tree), N };
	PyArrayObject * out_asgn = (PyArrayObject*) PyArray_NewFromDescr(
		&PyArray_Type, PyArray_DescrFromType(PyArray_INT32), 2, dims, NULL,
		NULL, NPY_F_CONTIGUOUS, NULL);

	vl_uint * asgn = (vl_uint *) out_asgn->data;

	vl_hikm_push(tree, asgn, data_ptr, N);

	for (int j = 0; j < N * depth; ++j)
		asgn[j]++;

	if (verb) {
		printf("hikmeans: done.\n");
	}

	object out_tree = hikm_to_python(tree);
	object agn_obj = object(handle<> ((PyObject*) out_asgn));
	return make_tuple(out_tree, agn_obj);
}








