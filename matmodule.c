/* Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
   For full license details, see the LICENSE file. */

/* This module is a placeholder - eventually, MATLAB's libmat 
   (for reading MATLAB's "mat" file format) should get an interface here. */
#define MATMODULE
#include "pymex.h"

static PyMethodDef mat_methods[] = {
  {NULL, NULL, 0, NULL}
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC initmatmodule(void) {
  PyObject *m = Py_InitModule3("matlab.mat", mat_methods, 
			       "MATLAB MAT-file interface (placeholder)");
  if (!m) return;

  matmodule = m;
}
