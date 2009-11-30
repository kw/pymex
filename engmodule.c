/* Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
   For full license details, see the LICENSE file. */

/* This module is a placeholder - eventually, MATLAB's Engine library
   (for spawning and talking to MATLAB processes) should get an interface here. */
#define ENGMODULE
#include "pymex.h"

static PyMethodDef eng_methods[] = {
  {NULL, NULL, 0, NULL}
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void 
#endif
PyMODINIT_FUNC initengmodule(void) {
  PyObject* m = Py_InitModule3("eng", eng_methods, 
			 "MATLAB Engine library, for spawning MATLAB processes. (placeholder)");
  if (!m) return;

  engmodule = m;
}
