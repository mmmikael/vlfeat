/*
 * vl_quickshift.h
 *
 * Author: Andreas Mueller
 *
 */

#include "../py_vlfeat.h"

extern "C" {
#include <vl/quickshift.h>
}

VlQS* vl_quickshift_new_python(PyArrayObject& image, int height, int width, int channels);

PyObject * vl_quickshift_get_dists_python(VlQS const *q);
PyObject * vl_quickshift_get_parents_python(VlQS const *q);
PyObject * vl_quickshift_get_density_python(VlQS const *q);
