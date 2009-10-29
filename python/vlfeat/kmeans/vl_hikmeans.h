/*
 * vl_hikmeans.h
 *
 *  Created on: Jul 13, 2009
 *      Author: Mikael Rousson
 */

#pragma once

#include "../py_vlfeat.h"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

void hikmeans_export();


class VlHIKMTree_python
{
public:
	VlHIKMTree_python(int _K, int _depth=0) : K(_K), depth(_depth){
		centers = NULL;
	}
	VlHIKMTree_python(const VlHIKMTree_python & tree) :
		K(tree.K), depth(tree.depth)
	{
		centers = tree.centers;
		sub = tree.sub;
		if (centers != NULL) {
			Py_INCREF(centers);
		}
	}
	~VlHIKMTree_python()
	{
		if (centers != NULL) {
			Py_DECREF(centers);
		}
	}

	bool operator==(const VlHIKMTree_python & t1)
	{
		return false;
	}

	PyObject * getCenters() const {
		return PyArray_Copy((PyArrayObject *) centers);
	}


	void save(char * file)
	{
		std::ofstream out(file, std::ofstream::binary);
		assert(out);
		write(out);
		out.close();
	}

	void write(std::ofstream & out) const {
		PyArrayObject * _centers = (PyArrayObject *) centers;
		int * centers_buffer = (int*) _centers->data;


		// write dimensions
		out.write((char *) &K, sizeof(int));
		out.write((char *) &depth, sizeof(int));
		out.write((char *) _centers->dimensions, sizeof(npy_intp) * 2);//_centers->nd);

		// write centers
		out.write(_centers->data, sizeof(int) *  PyArray_Size(centers));

		// write size
		int s = sub.size();
		out.write((char *) &s, sizeof(int));

		// write children
		for(int i=0; i<sub.size(); i++) {
			sub[i].write(out);
		}
	}

	void load(char * file)
	{
		std::ifstream in(file, std::ifstream::binary);
		assert(in);
		read(in);
		in.close();
	}

	void read(std::ifstream & in)
	{
		// read dimensions
		npy_intp dims[2];
		in.read((char *) &K, sizeof(int));
		in.read((char *) &depth, sizeof(int));
		in.read((char *) dims, sizeof(npy_intp) * 2);

		// allocate centers array
		centers = PyArray_NewFromDescr(
			&PyArray_Type, PyArray_DescrFromType(PyArray_INT32),
			2, dims, NULL, NULL, NPY_F_CONTIGUOUS, NULL);

		// read centers and fills array
		in.read((char *) (((PyArrayObject*) centers)->data), sizeof(int)
				* dims[0] * dims[1]);

		// read number of children
		int nbSub;
		in.read((char *) &nbSub, sizeof(int));

		// read children
		for (int i = 0; i < nbSub; i++) {
			sub.push_back(VlHIKMTree_python(K, 0));
			VlHIKMTree_python & psub = sub.back();
			psub.read(in);
		}
	}

	int K;
	int depth;
	PyObject * centers;
	std::vector<VlHIKMTree_python> sub;// tuple containing sub trees
};


/**
 *
 * @param data
 * @param K
 * @param nleaves
 * @param verb
 * @param max_iters
 * @param method
 * @return
 */
boost::python::tuple vl_hikmeans_python(
		PyArrayObject & data,
		int K,
		int nleaves,
		int verb,
		int max_iters,
		char * method);

/**
 *
 * @param inTree
 * @param inData
 * @param verb
 * @param method
 * @return
 */
PyObject * vl_hikmeanspush_python(
		VlHIKMTree_python & inTree,
		PyArrayObject & inData,
		int verb,
		char * method);
