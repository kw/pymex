/*
  The MATLAB Extension library module.
  This module (in theory) handles all communication with MATLAB originating
  on the python side. This is not actually true, since libmx will call into
  the MATLAB interpreter for certain things if it is available. Similarly,
  our matlab.mx library uses some libmex things when necessary. It calls into
  matlab.mex to generate string representations of objects, and directly calls
  mexMakeArrayPersistent when it creates things in mex mode. 
 */
/*
  FIXME: libmex is not threadsafe. Neither is CPython, but at least we have the GIL.
  MATLAB does not even have a concept of user-accessible independent threads, so it
  should be no surprise that they have no user-accessible locking mechanism. It might
  be appropriate to add a "MIL" to serialize libmex functions. Note that certain libmx
  functions call into libmex when compiled in mex mode, but which ones exactly? That
  appears to be undocumented, but we can make some educated guesses, and should modify
  matlab.mx accordingly if a MIL is implemented.
*/
#define MEXMODULE
#include "pymex.h"
#if MATLAB_MEX_FILE
#include <mex.h>

static PyObject* m_printf(PyObject* self, PyObject* args) {
  PyObject* format = PySequence_GetItem(args, 0);
  Py_ssize_t arglength = PySequence_Size(args);
  PyObject* tuple = PySequence_GetSlice(args, 1, arglength+1);
  PyObject* out = PyUnicode_Format(format, tuple);
  PyObject* b_out = PyUnicode_AsASCIIString(out);
  char* outstr = PyBytes_AsString(b_out);
  mexPrintf(outstr);
  Py_DECREF(out);
  Py_DECREF(b_out);
  Py_DECREF(tuple);
  Py_DECREF(format);
  Py_RETURN_NONE;
}

static PyObject* m_eval(PyObject* self, PyObject* args) {
  char* evalstring = NULL;
  if (!PyArg_ParseTuple(args, "s", &evalstring))
    return NULL;
  /* would prefer to use mexCallMATLABWithTrap, but not in 2008a */
  mexSetTrapFlag(1);
  mxArray* evalarray[2];
  evalarray[0] = mxCreateString("base");
  evalarray[1] = mxCreateString(evalstring);
  mxArray* out = NULL;
  int retval = mexCallMATLAB(1, &out, 2, evalarray, "evalin");
  mxDestroyArray(evalarray[0]);
  mxDestroyArray(evalarray[1]);
  /* TODO: Throw some sort of error instead. */
  if (retval)
    Py_RETURN_NONE;
  else {
    return Any_mxArray_to_PyObject(out);
  }
}

static PyObject* m_call(PyObject* self, PyObject* args, PyObject* kwargs) {
  static char *kwlist[] = {"nargout",NULL};
  int nargout = -1;
  PyObject* fakeargs = PyTuple_New(0);
  if (!PyArg_ParseTupleAndKeywords(fakeargs, kwargs, "|i", kwlist, &nargout))
    return NULL;
  Py_DECREF(fakeargs);
  int nargin = PySequence_Size(args);  
  mxArray *inargs[nargin];
  int i;
  for (i=0; i<nargin; i++) {
    inargs[i] = Any_PyObject_to_mxArray(PyTuple_GetItem(args, i));
  }
  int tupleout = nargout >= 0;
  if (nargout < 0) nargout = 1;
  mxArray *outargs[nargout];
  mexSetTrapFlag(1);
  int retval = mexCallMATLAB(nargout, outargs, nargin, inargs, "feval");
  if (retval)
    return PyErr_Format(PyExc_RuntimeError, "I have no idea what happened");
  else {
    if (tupleout) {
      PyObject* outseq = PyTuple_New(nargout);
      for (i=0; i<nargout; i++) {
	PyTuple_SetItem(outseq, i, Any_mxArray_to_PyObject(outargs[i]));
      }
      return outseq;
    }
    else
      return Any_mxArray_to_PyObject(outargs[0]);
  }
}

static PyMethodDef mex_methods[] = {
  {"printf", m_printf, METH_VARARGS, "Print a string using mexPrintf"},
  {"eval", m_eval, METH_VARARGS, "Evaluates a string using mexEvalString"},
  {"call", (PyCFunction)m_call, METH_VARARGS | METH_KEYWORDS, "feval the inputs"},
  {NULL, NULL, 0, NULL}
};
#else
static PyMethodDef mex_methods[] = {
  {NULL, NULL, 0, NULL}
};
#endif

#if PY_VERSION_HEX >= PY3K_VERSION_HEX
static PyModuleDef mexmodule_def = {
    PyModuleDef_HEAD_INIT,
    "matlab.mex",
    #if MATLAB_MEX_FILE
    "MATLAB Extension API module",
    #else
    "MATLAB Extension API module (only available inside MATLAB)",
    #endif
    -1,
    mex_methods, NULL, NULL, NULL, NULL
};
#endif

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC PyObject*
#endif
PyMODINIT_FUNC
initmexmodule(void)
{

  #if PY_VERSION_HEX < PY3K_VERSION_HEX
  PyObject* m = Py_InitModule3("matlab.mex", mex_methods, 
    #if MATLAB_MEX_FILE
    "MATLAB Extension API module"
    #else
    "MATLAB Extension API module (only available inside MATLAB)"
    #endif
			       );
  if (!m) return;
  #else
  PyObject* m = PyModule_Create(&mexmodule_def);
  if (!m) return NULL;
  #endif

  mexmodule = m;
  
  PyObject* sys = PyImport_AddModule("sys");
  PyObject* path = PyObject_GetAttrString(sys, "path");
  PyObject* pymexpath = PyObject_CallMethod(m, "eval", "s", "fileparts(which('pymex'));");
  if (PyList_Append(path, pymexpath) < 0) PyErr_Clear();

  #if PY_VERSION_HEX >= PY3K_VERSION_HEX
  return m;
  #endif
}
