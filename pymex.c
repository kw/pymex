#include <Python.h>
#include <numpy/arrayobject.h>
#include "mex.h"

#ifndef PYMEX_DEBUG_FLAG
#define PYMEX_DEBUG_FLAG 0
#endif

#if PYMEX_DEBUG_FLAG
#define PYMEX_DEBUG(format, args...) mexPrintf(format,##args)
#else
#define PYMEX_DEBUG(format, args...) /*nop*/
#endif



/* Macros used during x-macro expansion. */

#define PYMEX_SIG(name) \
void name##_pymexfun(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])

#define PYMEX_DEFINE(name, min, max, body)				\
  PYMEX_SIG(name) {							\
    PYMEX_DEBUG("<start " #name ">\n");					\
    if (nrhs < min || nrhs > max) {					\
      mexErrMsgIdAndTxt("pymex:" #name ":nargchk",			\
			"Bad number of args: %d <= %d <= %d",		\
			min, nrhs, max); }				\
    do body while (0);							\
    PYMEX_DEBUG("<end " #name ">\n");					\
  }

#define PYMEX_MAKECELL(name, min, max, body)	\
  mxSetCell(plhs[0], PYMEX_CMD_##name, mxCreateString(#name));

#define PYMEX_STRCMP(name, min, max, body)		\
  else if (!strcmp(#name, cmdstring)) {			\
    mxFree(cmdstring);					\
    name##_pymexfun(nlhs, plhs, nrhs-1, prhs+1);	\
  }

#define PYMEX_ENUM(name, min, max, body)	\
  PYMEX_CMD_##name,
  
#define PYMEX_CASE(name, min, max, body)		\
  case PYMEX_CMD_##name:				\
  name##_pymexfun(nlhs, plhs, nrhs-1, prhs+1);		\
  break;

/* Static utility stuff goes in here */
#include "pymex_static.c"

/* Define pymex command enums via x-macro */
#define PYMEX(name, min, max, body) PYMEX_ENUM(name,min,max,body)
enum PYMEX_COMMAND {
#include "pymex.def.c"
  NUMBER_OF_PYMEX_COMMANDS,
};
#undef PYMEX

/* Define pymex commands via x-macro */
#define PYMEX(name, min, max, body) PYMEX_DEFINE(name,min,max,body)
#include "pymex.def.c"
#undef PYMEX

/* mex body and related functions */

static void ExitFcn() {
  Py_Finalize();
  PYMEX_DEBUG("[python: finalized]\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (!Py_IsInitialized()) {
    Py_Initialize();
    PYMEX_DEBUG("[python: initialized]\n");
    mexAtExit(ExitFcn);
  }   

  if (nrhs < 1 || mxIsEmpty(prhs[0])) {
    if (nlhs == 1) {
      plhs[0] = mxCreateCellMatrix(1,NUMBER_OF_PYMEX_COMMANDS);
#define PYMEX(name,min,max,body) PYMEX_MAKECELL(name,min,max,body)
#include "pymex.def.c"
#undef PYMEX
    } 
    else {
      mexEvalString("help('pymex')");
    }
  }
  else if (mxIsNumeric(prhs[0])) {
    enum PYMEX_COMMAND cmd = (int) mxGetScalar(prhs[0]);

    /* Switch body defined via x-macro expansion */
    switch (cmd) {
#define PYMEX(name,min,max,body) PYMEX_CASE(name,min,max,body)
#include "pymex.def.c"
#undef PYMEX
    default:
      mexErrMsgIdAndTxt("pymex:NotImplemented", "pymex command %d not implemented", cmd);
    }
  }
  else if (mxIsChar(prhs[0])) {
    char* cmdstring = mxArrayToString(prhs[0]);
    if (!cmdstring) {
      mexErrMsgIdAndTxt("pymex:badstring", "Could not extract the command string for some reason.");
    } /* a bunch of else-ifs are generated here */
#define PYMEX(name,min,max,body) PYMEX_STRCMP(name,min,max,body)
#include "pymex.def.c"
#undef PYMEX
    else {
      mexErrMsgIdAndTxt("pymex:NotImplemented", "pymex command '%s' not implemented", cmdstring);
      /* mxFree(cmdstring) */
      /* We'd like to free it, but this won't run. Hopefully MATLAB's magic memory manager
         will handle this for us. If not, this shouldn't be a memory leak. Unless someone decides
         to write code that calls pymex directly, misspells the command string, catchs and ignores
         the error, and then somehow doesn't notice that whatever they're doing isn't working.
      */
    }
  }
  else {
    mexErrMsgIdAndTxt("pymex:badcmd", "I don't really know what to do with a %s", mxGetClassName(prhs[0]));
  }

  /* Detect and pass on python errors */
  PyObject* err = PyErr_Occurred();
  if (err) {
    PyObject *err_type, *err_value, *err_traceback;
    PyErr_Fetch(&err_type, &err_value, &err_traceback);
    /*mexPutVariable("global", "PYMEX_ERR_TYPE", box(err_type));
    mexPutVariable("global", "PYMEX_ERR_VALUE", box(err_value));
    mexPutVariable("global", "PYMEX_ERR_TRACE", box(err_traceback));*/
    if (!err_value)
      err_value = PyString_FromString("<no value>");
    PyObject* pyid = PyString_FromString("Python:"); 
    PyString_ConcatAndDel(&pyid, PyObject_GetAttrString(err_type, "__name__"));
    PyObject* tuple = Py_BuildValue("ON", pyid, PyObject_Str(err_value));
    PyObject* format = PyString_FromString("%s -> %s\n");
    PyObject* pymsg = PyString_Format(format, tuple);
    char* id = PyString_AsString(pyid);
    char* msg = PyString_AsString(pymsg);
    Py_DECREF(pyid);
    Py_DECREF(tuple);
    Py_DECREF(format);
    Py_DECREF(pymsg);
    PyErr_Clear();
    mexErrMsgIdAndTxt(id, msg);
  }
}

