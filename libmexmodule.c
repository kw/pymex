#include <Python.h>
#include "structmember.h"
#define PY_ARRAY_UNIQUE_SYMBOL PYMEX_ARRAY_API
#define NPY_USE_PYMEM 1
#include <numpy/arrayobject.h>
#define LIBMEXMODULE
#include "pymex.h"
#include <mex.h>

static PyObject* m_printf(PyObject* self, PyObject* args) {
  PyObject* format = PySequence_GetItem(args, 0);
  Py_ssize_t arglength = PySequence_Size(args);
  PyObject* tuple = PySequence_GetSlice(args, 1, arglength+1);
  PyObject* out = PyString_Format(format, tuple);
  char* outstr = PyString_AsString(out);
  mexPrintf(outstr);
  Py_DECREF(out);
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
  int nargout = 1;
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
  mxArray *outargs[nargout];
  mexSetTrapFlag(1);
  int retval = mexCallMATLAB(nargout, outargs, nargin, inargs, "feval");
  /* TODO: Throw some sort of error instead. */
  if (retval)
    Py_RETURN_NONE;
  else {
    PyObject* outseq = PyTuple_New(nargout);
    for (i=0; i<nargout; i++) {
      PyTuple_SetItem(outseq, i, Any_mxArray_to_PyObject(outargs[i]));
    }
    return outseq;
  }
}

static PyMethodDef libmex_methods[] = {
  {"printf", m_printf, METH_VARARGS, "Print a string using mexPrintf"},
  {"eval", m_eval, METH_VARARGS, "Evaluates a string using mexEvalString"},
  {"call", (PyCFunction)m_call, METH_VARARGS | METH_KEYWORDS, "feval the inputs"},
  {NULL, NULL, 0, NULL}
};

/* libmx mxArray type */

static void 
mxArray_dealloc(mxArrayObject* self)
{
  mxArray* ptr = mxArrayPtr(self);
  if (ptr) mxDestroyArray(ptr);
  self->ob_type->tp_free((PyObject*) self);
}

static PyObject*
mxArray_mxclass_id(mxArrayObject* self)
{
  mxClassID id = mxGetClassID(mxArrayPtr(self));
  return PyInt_FromLong((long) id);
}

static PyObject*
mxArray_mxclass(mxArrayObject* self)
{
  const char* class = mxGetClassName(mxArrayPtr(self));
  return PyString_FromString(class);
}

static PyMethodDef mxArray_methods[] = {
  {"mxGetClassID", (PyCFunction)mxArray_mxclass_id, METH_NOARGS,
   "Returns the ClassId of the mxArray"},
  {"mxGetClassName", (PyCFunction)mxArray_mxclass, METH_NOARGS,
   "Returns the name of the class of the mxArray"},
  {NULL}
};

static PyTypeObject mxArrayType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "libmex.mxArray",          /*tp_name*/
    sizeof(mxArrayObject),    /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)mxArray_dealloc,    /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    "mxArray objects",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    mxArray_methods,             /* tp_methods */
    0,                        /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    0,                        /* tp_new */
};



#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initlibmexmodule(void)
{
  mxArrayType.tp_new = PyType_GenericNew;
  if (PyType_Ready(&mxArrayType) < 0) return;

  libmexmodule = Py_InitModule("libmex", libmex_methods);

  Py_INCREF(&mxArrayType);
  PyModule_AddObject(libmexmodule, "mxArray", (PyObject*) &mxArrayType);

  /* numpy C-API import */
  import_array();
}
