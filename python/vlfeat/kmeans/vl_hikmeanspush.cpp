/** @internal
 ** @file    hikmeanspush.c
 ** @brief   vl_hikm_push - MEX driver
 ** @author  Brian Fulkerson
 ** @author  Andrea Vedaldi
 ** @author  Mikael Rousson (Python wrapping)
 **/

/* AUTORIGHTS
 Copyright 2007 (c) Andrea Vedaldi and Brian Fulkerson

 This file is part of VLFeat, available in the terms of the GNU
 General Public License version 2.
 */

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

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Convert C++/Python structure to HIKM node
 **/

static VlHIKMNode *
xcreate(VlHIKMTree *tree, VlHIKMTree_python & inTree)
{
	VlHIKMNode *node;
	int M, node_K, k;

	PyArrayObject * centers = (PyArrayObject *) inTree.centers;
	M = centers->dimensions[0];
	node_K = centers->dimensions[1];

	if (node_K > tree->K) {
		printf("%d / %d\n", node_K, tree->K);
		printf("A node has more clusters than TREE.K.\n");
	}

	if (tree->M < 0) {
		tree->M = M;
	} else if (M != tree->M) {
		printf("%d / %d\n", M, tree->M);
		printf("A node CENTERS field has inconsistent dimensionality.\n");
	}

	node = new VlHIKMNode;
	node->filter = vl_ikm_new(tree->method);
	node->children = 0;

	vl_ikm_init(node->filter, (vl_ikm_acc*) centers->data, M, node_K);

	/* has any childer? */
	if (inTree.sub.size() > 0) {

		/* sanity checks */
		if (inTree.sub.size() != node_K) {
			printf("%d, %d\n", inTree.sub.size(), node_K);
			printf("NODE.SUB size must correspond to NODE.CENTERS.\n");
		}

		node-> children = new VlHIKMNode*[node_K];
		for (k = 0; k < node_K; ++k) {
			PyArrayObject * centers = (PyArrayObject *) inTree.sub[k].centers;
			node-> children[k] = xcreate(tree, inTree.sub[k]);
		}
	}
	return node;
}

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Convert C++/Python structure to HIKM tree
 **/

static VlHIKMTree*
python_to_hikm(VlHIKMTree_python & inTree, int method_type)
{
	VlHIKMTree *tree = new VlHIKMTree;
	tree-> depth = inTree.depth;
	tree-> K = inTree.K;
	tree-> M = -1; /* to be initialized later */
	tree-> method = method_type;
	tree-> root = xcreate(tree, inTree);
	return tree;
}

/** ----------------------------------------------------------------
 **
 **/
PyObject * vl_hikmeanspush_python(
		VlHIKMTree_python & inTree,
		PyArrayObject & inData,
		int verb,
		char * method)
{
	vl_uint8 const *data;

	int N = 0;
	int method_type = VL_IKM_LLOYD;

	N = inData.dimensions[1]; /* n of elements */

#ifdef DEBUG
	printf("n of elements: %d\n", N);
	printf("n of split: %d\n", inTree.K);
	printf("depth: %d\n", inTree.depth);
	printf("n of children: %d\n", inTree.sub.size());
#endif

	data = (vl_uint8 *) inData.data;

	if (strcmp("lloyd", method) == 0) {
		method_type = VL_IKM_LLOYD;
	} else if (strcmp("elkan", method) == 0) {
		method_type = VL_IKM_ELKAN;
	} else {
		printf("Unknown cost type.\n");
	}

	/* -----------------------------------------------------------------
	 *                                                        Do the job
	 * -------------------------------------------------------------- */

	VlHIKMTree * tree;
	vl_uint *ids;
	int j;
	int depth;

	tree = python_to_hikm(inTree, method_type);
	depth = vl_hikm_get_depth(tree);

	if (verb) {
		printf("hikmeanspush: ndims: %d K: %d depth: %d\n", vl_hikm_get_ndims(
			tree), vl_hikm_get_K(tree), depth);
	}

	npy_intp dims[2] = { depth, N };
	PyArrayObject * out_asgn = (PyArrayObject*) PyArray_NewFromDescr(
		&PyArray_Type, PyArray_DescrFromType(PyArray_INT32), 2, dims, NULL,
		NULL, NPY_F_CONTIGUOUS, NULL);

	ids = (vl_uint *) out_asgn->data;

	vl_hikm_push(tree, ids, data, N);
	vl_hikm_delete(tree);

	for (j = 0; j < N * depth; j++)
		ids[j]++;

	return PyArray_Return(out_asgn);

}



