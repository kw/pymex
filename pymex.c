#include <Python.h>
#include "mex.h"

#define PYMEX_DEBUG 0

typedef unsigned long long UINT64;

/* Macros used during x-macro expansion. */

#define PYMEX_SIG(name) \
static void pymex_##name (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])

#if PYMEX_DEBUG
#define PYMEX_DEFINE(name, number, body)	\
  PYMEX_SIG(name) {				\
    mexPrintf("<start %s>\n", #name);		\
    do body while (0);				\
    mexPrintf("<end %s>\n", #name);}
#else
#define PYMEX_DEFINE(name, number, body)	\
  PYMEX_SIG(name) body
#endif

#define PYMEX_CASE(name,number,body)	    \
  case number:				    \
  pymex_##name(nlhs, plhs, nrhs-1, prhs+1); \
  break;

/* Static utility stuff goes in here */
#include "pymex_static.c"

/* Define pymex commands via x-macro */
#define PYMEX(name, number, body) PYMEX_DEFINE(name,number,body)
#include "pymex.def.c"
#undef PYMEX

/* mex body and related functions */

static void ExitFcn() {
  Py_Finalize();
  mexPrintf("[python: finalized]\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 1 || mxIsEmpty(prhs[0]) || !mxIsNumeric(prhs[0])) {
    mexErrMsgIdAndTxt("pymex:badarg", "First argument must be a numeric command from py.Interface");
  }

  int cmd = (int) mxGetScalar(prhs[0]);

  if (!Py_IsInitialized()) {
    Py_Initialize();
    mexPrintf("[python: initialized]\n");
    mexAtExit(ExitFcn);
  }  

  /* Switch body defined via x-macro expansion */
  switch (cmd) {
#define PYMEX(name,number,body) PYMEX_CASE(name,number,body)
#include "pymex.def.c"
#undef PYMEX
  default:
    mexErrMsgIdAndTxt("pymex:NotImplemented", "pymex command %d not implemented", cmd);
  }

  PyObject* err = PyErr_Occurred();
  if (err) {
    PyObject *err_type, *err_value, *err_traceback;
    PyErr_Fetch(&err_type, &err_value, &err_traceback);
    mexPutVariable("global", "PYMEX_ERR_TYPE", box(err_type));
    mexPutVariable("global", "PYMEX_ERR_VALUE", box(err_value));
    mexPutVariable("global", "PYMEX_ERR_TRACE", box(err_traceback));
    if (err_value && PyString_Check(err_value)) {
      char* msg = PyString_AsString(err_value);
      mexErrMsgIdAndTxt("pymex:PythonError", "Python said: %s\n(See global PYMEX_ERR_*)", msg);
    }
    else {
      mexErrMsgIdAndTxt("pymex:PythonError", "Error in Python, but wasn't a string value."
			" See global PYMEX_ERR_*");
    }
    PyErr_Clear();
  }
}

