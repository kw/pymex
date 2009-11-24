/* Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
   For full license details, see the LICENSE file. */

/* This module is a placeholder - eventually, MATLAB's libmat 
   (for reading MATLAB's "mat" file format) should get an interface here. */
#define MATMODULE
#include "pymex.h"

static PyMethodDef mat_methods[] = {
  {NULL, NULL, 0, NULL}
};

#if PY_VERSION_HEX >= PY3K_VERSION_HEX
static PyModuleDef matmodule_def = {
    PyModuleDef_HEAD_INIT,
    "matlab.mat",
    "MATLAB MAT-file interface (placeholder)",
    -1,
    mat_methods, NULL, NULL, NULL, NULL
};
#endif

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC PyObject*
#endif
PyMODINIT_FUNC
initmatmodule(void)
{

  #if PY_VERSION_HEX < PY3K_VERSION_HEX
  PyObject* m = Py_InitModule3("matlab.mat", mat_methods, "MATLAB MAT-file interface (placeholder)");
  if (!m) return;
  #else
  PyObject* m = PyModule_Create(&matmodule_def);
  if (!m) return NULL;
  #endif

  matmodule = m;

  #if PY_VERSION_HEX >= PY3K_VERSION_HEX
  return m;
  #endif
}
