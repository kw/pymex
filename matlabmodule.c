/* Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
   For full license details, see the LICENSE file. */

#include <Python.h>
#include "structmember.h"
#define MATLABMODULE
#include "pymex.h"

static PyMethodDef matlab_methods[] = {
  {NULL, NULL, 0, NULL}
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC initmatlabmodule(void) {
  PyObject *m = Py_InitModule3("matlab", matlab_methods, "MATLAB C-API module container");
  if (!m) return;

  MATLABError = PyErr_NewException("matlab.MATLABError", NULL, NULL);
  Py_INCREF(MATLABError);

  matlabmodule = m;

  initmexmodule();
  Py_INCREF(mexmodule);
  PyModule_AddObject(m, "mex", mexmodule);

  initmxmodule();
  Py_INCREF(mxmodule);
  PyModule_AddObject(m, "mx", mxmodule);

  initmatmodule();
  Py_INCREF(matmodule);
  PyModule_AddObject(m, "mat", matmodule);
  
  initengmodule();
  Py_INCREF(engmodule);
  PyModule_AddObject(m, "eng", engmodule);
}
