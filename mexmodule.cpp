/* Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
   For full license details, see the LICENSE file. */

/*
  The MATLAB Extension library module.
  This module (in theory) handles all communication with MATLAB originating
  on the python side. This is not actually true, since libmx will call into
  the MATLAB interpreter for certain things if it is available. Similarly,
  our matlab.mx library uses some libmex things when necessary. It calls into
  matlab.mex to generate string representations of objects, and directly calls
  mexMakeArrayPersistent when it creates things in mex mode. 
 */

#define MEXMODULE
#include "pymex.h"
#if MATLAB_MEX_FILE
#include <mex.h>

static PyObject *m_printf(PyObject *self, PyObject *args) {
  PyObject *format = PySequence_GetItem(args, 0);
  Py_ssize_t arglength = PySequence_Size(args);
  PyObject *tuple = PySequence_GetSlice(args, 1, arglength+1);
  PyObject *out = PyUnicode_Format(format, tuple);
  PyObject *b_out = PyUnicode_AsASCIIString(out);
  char *outstr = PyBytes_AsString(b_out);
  mexPrintf(outstr);
  Py_DECREF(out);
  Py_DECREF(b_out);
  Py_DECREF(tuple);
  Py_DECREF(format);
  Py_RETURN_NONE;
}

static PyObject *_raiselasterror(PyObject *self) {
  mxArray *argin;
  mxArray *argout;
  argin = mxCreateString("reset");
  mxArray *err = mexCallMATLABWithTrap(1, &argout, 1, &argin, "lasterror");
  if (!err) {
    char *id = mxArrayToString(mxGetField(argout, 0, "identifier"));
    char *msg = mxArrayToString(mxGetField(argout, 0, "message"));
    PyObject *stack = Any_mxArray_to_PyObject(mxGetField(argout, 0, "stack"));
    PyObject *errval = Py_BuildValue("(ssO)", id, msg, stack);
    PyErr_SetObject(MATLABError, errval);
    mxFree(id);
    mxFree(msg);
    Py_DECREF(stack);
    Py_DECREF(errval); /* Docs don't mention it, but apparently
			  PyErr_SetObject doesn't steal the reference. */
  }
  else {
    PyErr_SetString(MATLABError, "MATLAB Error occurred, "
		    "but could not retrieve error struct.");
  }
  return NULL;
}

static PyObject *m_eval(PyObject *self, PyObject *args) {
  char *evalstring = NULL;
  if (!PyArg_ParseTuple(args, "s", &evalstring))
    return NULL;
  mxArray *evalarray[2];
  evalarray[0] = mxCreateString("base");
  evalarray[1] = mxCreateString(evalstring);
  mxArray *out = NULL;
  mxArray *err = mexCallMATLABWithTrap(1, &out, 2, evalarray, "evalin");
  mxDestroyArray(evalarray[0]);
  mxDestroyArray(evalarray[1]);
  if (err)
    return _raiselasterror(NULL);
  else {
    return Any_mxArray_to_PyObject(out);
  }
}

static PyObject *m_call(PyObject *self, PyObject *args, PyObject *kwargs) {
  static char *kwlist[] = {"nargout","wrap",NULL};
  int nargout = -1;
  int wrap = 1;
  PyObject *fakeargs = PyTuple_New(0);
  if (!PyArg_ParseTupleAndKeywords(fakeargs, kwargs, "|ii", kwlist, &nargout,&wrap))
    return NULL;
  Py_DECREF(fakeargs);
  int nargin = PySequence_Size(args);  
  mxArray **inargs;
  inargs = new mxArray*[nargin];
  int i;
  for (i=0; i<nargin; i++) {
    inargs[i] = Any_PyObject_to_mxArray(PyTuple_GetItem(args, i));
  }
  delete [] inargs;
  int tupleout = nargout >= 0;
  if (nargout < 0) nargout = 1;
  mxArray **outargs;
  outargs = new mxArray*[nargout];
  mxArray *err = mexCallMATLABWithTrap(nargout, outargs, 
				       nargin, inargs, "feval");
  if (err) {
    delete [] outargs;
    return _raiselasterror(NULL);
  }
  else {
    if (tupleout) {
      PyObject *outseq = PyTuple_New(nargout);
      for (i=0; i<nargout; i++) {
	PyTuple_SetItem(outseq, i, 
			wrap 
			? Any_mxArray_to_PyObject(outargs[i]) 
			: mxArrayPtr_New(outargs[i])
			);
      }
      delete [] outargs;
      return outseq;
    }
    else {
      mxArray *outzero = outargs[0];
      delete [] outargs;
      return wrap 
	? Any_mxArray_to_PyObject(outzero)
	: mxArrayPtr_New(outzero);
    }
  }
}

static PyMethodDef mex_methods[] = {
  {"printf", m_printf, METH_VARARGS, "Print a string using mexPrintf"},
  {"eval", m_eval, METH_VARARGS, "Evaluates a string using mexEvalString"},
  {"call", (PyCFunction)m_call, METH_VARARGS | METH_KEYWORDS, "feval the inputs"},
  {"__raiselasterror", (PyCFunction)_raiselasterror, METH_NOARGS,
   "Raises a MATLABError. Attempts to retrieve the MATLAB error struct to do so."},
  {NULL, NULL, 0, NULL}
};
#else
static PyMethodDef mex_methods[] = {
  {NULL, NULL, 0, NULL}
};
#endif


#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC initmexmodule(void) {
  PyObject *m = Py_InitModule3("mex", mex_methods, 
    "MATLAB Extension API module (only available inside MATLAB)"
			       );
  if (!m) return;

  mexmodule = m;
  
  PyObject *sys = PyImport_AddModule("sys");
  PyObject *path = PyObject_GetAttrString(sys, "path");
  PyObject *pymexpath = PyObject_CallMethod(m, "eval", "s", 
					    "fileparts(which('pymex'));");
  if (PyList_Append(path, pymexpath) < 0) PyErr_Clear();
  PyObject *argv = PyList_New(1);
  PyObject *arg0 = PyBytes_FromString("matlab");
  PyList_SetItem(argv, 0, arg0);
  Py_DECREF(arg0);
  if (PyModule_AddObject(sys, "argv", argv) < 0) PyErr_Clear();
}
