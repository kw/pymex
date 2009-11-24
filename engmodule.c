/* Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
   For full license details, see the LICENSE file. */

/* This module is a placeholder - eventually, MATLAB's Engine library
   (for spawning and talking to MATLAB processes) should get an interface here. */
#define ENGMODULE
#include "pymex.h"

static PyMethodDef eng_methods[] = {
  {NULL, NULL, 0, NULL}
};

#if PY_VERSION_HEX >= PY3K_VERSION_HEX
static PyModuleDef engmodule_def = {
    PyModuleDef_HEAD_INIT,
    "matlab.eng",
    "MATLAB Engine library, for spawning MATLAB processes. (placeholder)"
    -1,
    eng_methods, NULL, NULL, NULL, NULL
};
#endif

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC PyObject*
#endif
PyMODINIT_FUNC
initengmodule(void)
{

  #if PY_VERSION_HEX < PY3K_VERSION_HEX
  PyObject* m = Py_InitModule3("matlab.eng", eng_methods, "MATLAB Engine library, for spawning MATLAB processes. (placeholder)");
  if (!m) return;
  #else
  PyObject* m = PyModule_Create(&engmodule_def);
  if (!m) return NULL;
  #endif

  engmodule = m;

  #if PY_VERSION_HEX >= PY3K_VERSION_HEX
  return m;
  #endif
}
