/**
 ** @author   Andrea Vedaldi
 ** @author   Mikael Rousson (Python wrapping)
 ** @brief    Scale Invariant Feature Transform (SIFT) - Python wrapper
 **/

#include "../py_vlfeat.h"

extern "C" {
#include <vl/mathop.h>
#include <vl/sift.h>
}

#include <math.h>
#include <assert.h>

#include <iostream>

/** ------------------------------------------------------------------
 ** @internal
 ** @brief Transpose desriptor
 **
 ** @param dst destination buffer.
 ** @param src source buffer.
 **
 ** The function writes to @a dst the transpose of the SIFT descriptor
 ** @a src. The tranpsose is defined as the descriptor that one
 ** obtains from computing the normal descriptor on the transposed
 ** image.
 **/

VL_INLINE void transpose_descriptor(vl_sift_pix* dst, vl_sift_pix* src)
{
	int const BO = 8; /* number of orientation bins */
	int const BP = 4; /* number of spatial bins     */
	int i, j, t;

	for (j = 0; j < BP; ++j) {
		int jp = BP - 1 - j;
		for (i = 0; i < BP; ++i) {
			int o = BO * i + BP * BO * j;
			int op = BO * i + BP * BO * jp;
			dst[op] = src[o];
			for (t = 1; t < BO; ++t)
				dst[BO - t + op] = src[t + o];
		}
	}
}

/** -------------------------------------------------------------------
 ** @internal
 ** @brief Ordering of tuples by increasing scale
 **
 ** @param a tuple.
 ** @param b tuble.
 **
 ** @return @c a[2] < b[2]
 **/

static int korder(void const* a, void const* b)
{
	double x = ((double*) a)[2] - ((double*) b)[2];
	if (x < 0)
		return -1;
	if (x > 0)
		return +1;
	return 0;
}

/** -------------------------------------------------------------------
 ** @internal
 ** @brief Check for sorted keypoints
 **
 ** @param keys keypoint list to check
 ** @param nkeys size of the list.
 **
 ** @return 1 if the keypoints are storted.
 **/

int check_sorted(double const * keys, int unsigned nkeys)
{
	int k;
	for (k = 0; k < nkeys - 1; ++k) {
		if (korder(keys, keys + 4) > 0) {
			return 0;
		}
		keys += 4;
	}
	return 1;
}

/** ------------------------------------------------------------------
 ** @brief Python entry point
 **/
PyObject * vl_sift_python(
		PyArrayObject & image,
		PyArrayObject & opt_frames,
		int opt_octaves,
		int opt_levels,
		int opt_first_octave,
		double opt_peak_thresh,
		double opt_edge_thresh,
		double opt_norm_thresh,
		double opt_magnif,
		double opt_window_size,
		bool opt_orientations,
		int opt_verbose)
{
	// check types
	assert(image.descr->type_num == PyArray_FLOAT);
	assert(image.flags & NPY_FORTRAN);

	// check if param values are valid
	assert(opt_octaves==-1 || opt_octaves>=0);
	assert(opt_levels==-1 || opt_levels>=0);
	assert(opt_first_octave==-1 || opt_first_octave>=0);
	assert(opt_peak_thresh==-1 || opt_peak_thresh>=0);
	assert(opt_edge_thresh==-1 || opt_edge_thresh>=1);
	assert(opt_norm_thresh==-1 || opt_norm_thresh>=0);
	assert(opt_magnif==-1 || opt_magnif>=0);

	enum
	{
		IN_I = 0, IN_END
	};
	enum
	{
		OUT_FRAMES = 0, OUT_DESCRIPTORS
	};

	int verbose = opt_verbose;
	int opt;
	int next = IN_END;

	vl_sift_pix const *data;
	int M, N;

	int O = -1;
	int S = 3;
	int o_min = 0;
	// set optional parameters
	if (opt_octaves >= 0) O = opt_octaves;
	if (opt_levels >= 0) S = opt_levels;
	o_min = opt_first_octave;

	double edge_thresh = opt_edge_thresh;
	double peak_thresh = opt_peak_thresh;
	double norm_thresh = opt_norm_thresh;
	double magnif = opt_magnif;
	double window_size = opt_window_size;

	double *ikeys = 0;
	int nikeys = -1;
	vl_bool force_orientations = opt_orientations;

	// create Python objects for outputs
	PyObject * _frames;
	PyObject * _descriptors;

	// get data and dims
	data = (vl_sift_pix*) image.data;
	M = image.dimensions[0];
	N = image.dimensions[1];


	// get input frames and sort them w.r.t scale
	if (opt_frames.nd > 1)
	{
		assert(opt_frames.descr->type_num == PyArray_FLOAT64);
		assert(opt_frames.flags & NPY_FORTRAN);
		ikeys = (double *) opt_frames.data;
		nikeys = opt_frames.dimensions[1];
		if (!check_sorted(ikeys, nikeys)) {
			qsort(ikeys, nikeys, 4 * sizeof(double), korder);
		}
	}

	/* -----------------------------------------------------------------
	 *                                                            Do job
	 * -------------------------------------------------------------- */
	{
		VlSiftFilt *filt;
		vl_bool first;
		double *frames = NULL;
		vl_uint8 *descr = NULL;

		int nframes = 0, reserved = 0, i, j, q;

		/* create a filter to process the image */
		filt = vl_sift_new(M, N, O, S, o_min);

		if (peak_thresh >= 0)
			vl_sift_set_peak_thresh(filt, peak_thresh);
		if (edge_thresh >= 0)
			vl_sift_set_edge_thresh(filt, edge_thresh);
		if (norm_thresh >= 0)
			vl_sift_set_norm_thresh(filt, norm_thresh);
		if (magnif >= 0)
			vl_sift_set_magnif(filt, magnif);
		if (window_size >= 0)
			vl_sift_set_window_size(filt, window_size);

		if (verbose) {
			printf("siftmx: filter settings:\n");
			printf(
				"siftmx:   octaves      (O)      = %d\n", vl_sift_get_noctaves(
					filt));
			printf(
				"siftmx:   levels       (S)      = %d\n", vl_sift_get_nlevels(
					filt));
			printf(
				"siftmx:   first octave (o_min)  = %d\n",
				vl_sift_get_octave_first(filt));
			printf(
				"siftmx:   edge thresh           = %g\n",
				vl_sift_get_edge_thresh(filt));
			printf(
				"siftmx:   peak thresh           = %g\n",
				vl_sift_get_peak_thresh(filt));
			printf(
				"siftmx:   norm thresh           = %g\n",
				vl_sift_get_norm_thresh(filt));
			printf(
				"siftmx:   magnif                = %g\n", vl_sift_get_magnif(
					filt));
			printf(
				"siftmx:   window size           = %g\n",
				vl_sift_get_window_size(filt));
			printf(
				(nikeys >= 0) ? "siftmx: will source frames? yes (%d read)\n"
						: "siftmx: will source frames? no\n", nikeys);
			printf(
				"siftmx: will force orientations? %s\n",
				force_orientations ? "yes" : "no");

		}

		/* ...............................................................
		 *                                             Process each octave
		 * ............................................................ */
		i = 0;
		first = 1;
		while (1) {
			int err;
			VlSiftKeypoint const *keys = 0;
			int nkeys = 0;

			if (verbose) {
				printf(
					"siftmx: processing octave %d\n", vl_sift_get_octave_index(
						filt));
			}

			/* Calculate the GSS for the next octave .................... */
			if (first) {
				err = vl_sift_process_first_octave(filt, data);
				first = 0;
			} else {
				err = vl_sift_process_next_octave(filt);
			}

			if (err)
				break;

//			if (verbose) {
//				printf(
//					"siftmx: GSS octave %d computed\n",
//					vl_sift_get_octave_index(filt));
//			}

			/* Run detector ............................................. */
			if (nikeys < 0) {
				vl_sift_detect(filt);

				keys = vl_sift_get_keypoints(filt);
				nkeys = vl_sift_get_nkeypoints(filt);
				i = 0;

				if (verbose) {
					printf(
						"siftmx: detected %d (unoriented) keypoints\n", nkeys);
				}
			} else {
				nkeys = nikeys;
			}

			/* For each keypoint ........................................ */
			for (; i < nkeys; ++i) {
				double angles[4];
				int nangles;
				VlSiftKeypoint ik;
				VlSiftKeypoint const *k;

				/* Obtain keypoint orientations ........................... */
				if (nikeys >= 0) {
					vl_sift_keypoint_init(
						filt, &ik,
						ikeys[4 * i + 1],
						ikeys[4 * i + 0],
						ikeys[4 * i + 2]);

					if (ik.o != vl_sift_get_octave_index(filt)) {
						break;
					}

					k = &ik;

					/* optionally compute orientations too */
					if (force_orientations) {
						nangles = vl_sift_calc_keypoint_orientations(
							filt, angles, k);
					} else {
						angles[0] = VL_PI / 2 - ikeys[4 * i + 3];
						nangles = 1;
					}
				} else {
					k = keys + i;
					nangles = vl_sift_calc_keypoint_orientations(
						filt, angles, k);
				}

				/* For each orientation ................................... */
				for (q = 0; q < nangles; ++q) {
					vl_sift_pix buf[128];
					vl_sift_pix rbuf[128];

					/* compute descriptor (if necessary) */
					//if (nout > 1) {


					vl_sift_calc_keypoint_descriptor(filt, buf, k, angles[q]);
					transpose_descriptor(rbuf, buf);

					//}
					/* make enough room for all these keypoints and more */
					if (reserved < nframes + 1) {
						reserved += 2 * nkeys;
						frames = (double *) realloc(frames, 4 * sizeof(double)
								* reserved);
						descr = (vl_uint8 *) realloc(descr, 128
								* sizeof(vl_uint8) * reserved);
					}

					/* Save back with MATLAB conventions. Notice that the input
					 * image was the transpose of the actual image. */
					frames[4 * nframes + 0] = k -> y;
					frames[4 * nframes + 1] = k -> x;
					frames[4 * nframes + 2] = k -> sigma;
					frames[4 * nframes + 3] = VL_PI / 2 - angles[q];

					for (j = 0; j < 128; ++j) {
						double x = 512.0 * rbuf[j];
						x = (x < 255.0) ? x : 255.0;
						descr[128 * nframes + j] = (vl_uint8) (x);
					}
					//}

					++nframes;
				} /* next orientation */
			} /* next keypoint */
		} /* next octave */

		if (verbose) {
			printf("siftmx: found %d keypoints\n", nframes);
		}

		/* ...............................................................
		 *                                                       Save back
		 * ............................................................ */

		// allocate pyarray objects (column-majored)
		npy_intp dims[2];
		dims[0] = 4;
		dims[1] = nframes;

		// frames
		_frames = PyArray_NewFromDescr(
			&PyArray_Type, PyArray_DescrFromType(PyArray_DOUBLE),
			2, dims, NULL, frames, NPY_F_CONTIGUOUS, NULL);
		PyArray_FLAGS(_frames) |= NPY_OWNDATA;

		// descriptors
		dims[0] = 128;
		_descriptors = PyArray_NewFromDescr(
			&PyArray_Type, PyArray_DescrFromType(PyArray_UBYTE),
			2, dims, NULL, (char*)descr, NPY_F_CONTIGUOUS, NULL);
		PyArray_FLAGS(_descriptors) |= NPY_OWNDATA;

		/* cleanup */
		vl_sift_delete(filt);

	} /* end: do job */

	// construct tuple to return both results: (regions, frames)
	PyObject * tuple = PyTuple_New(2);
	PyTuple_SetItem(tuple, 0, _frames);
	PyTuple_SetItem(tuple, 1, _descriptors);

	return tuple;
}
