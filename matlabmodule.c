#include <Python.h>
#include "structmember.h"
#if PYMEX_USE_NUMPY
 #define PY_ARRAY_UNIQUE_SYMBOL PYMEX_ARRAY_API
 #define NPY_USE_PYMEM 1
 #include <numpy/arrayobject.h>
#endif
#define MATLABMODULE
#include "pymex.h"

static PyMethodDef matlab_methods[] = {
  {NULL, NULL, 0, NULL}
};

#if PY_VERSION_HEX >= PY3K_VERSION_HEX
static PyModuleDef matlabmodule_def = {
    PyModuleDef_HEAD_INIT,
    "matlab",
    "MATLAB C-API module container",
    -1,
    matlab_methods, NULL, NULL, NULL, NULL
};
#endif

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC PyObject*
#endif
PyMODINIT_FUNC
initmatlabmodule(void)
{

  #if PY_VERSION_HEX < PY3K_VERSION_HEX
  PyObject* m = Py_InitModule3("matlab", matlab_methods, "MATLAB C-API module container");
  if (!m) return;
  #else
  PyObject* m = PyModule_Create(&matlabmodule_def);
  if (!m) return NULL;
  #endif

  #if PYMEX_USE_NUMPY
  /* numpy C-API import */
  import_array();
  #endif

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

  #if PY_VERSION_HEX >= PY3K_VERSION_HEX
  return m;
  #endif
}
