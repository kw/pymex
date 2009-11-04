#include <Python.h>
#include "structmember.h"
#define PY_ARRAY_UNIQUE_SYMBOL PYMEX_ARRAY_API
#define NPY_USE_PYMEM 1
#include <numpy/arrayobject.h>
#define LIBMEXMODULE
#include "pymex.h"
#include <mex.h>
#include <signal.h>

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

static PyMethodDef libmex_methods[] = {
  {"printf", m_printf, METH_VARARGS, "Print a string using mexPrintf"},
  {"eval", m_eval, METH_VARARGS, "Evaluates a string using mexEvalString"},
  {"call", (PyCFunction)m_call, METH_VARARGS | METH_KEYWORDS, "feval the inputs"},
  {NULL, NULL, 0, NULL}
};

/* libmx mxArray type */

static void 
mxArray_dealloc(PyObject* self)
{
  mxArray* ptr = mxArrayPtr(self);
  if (ptr) mxDestroyArray(ptr);
  self->ob_type->tp_free((PyObject*) self);
}

static PyObject*
mxArray_mxGetClassID(PyObject* self)
{
  mxClassID id = mxGetClassID(mxArrayPtr(self));
  return PyInt_FromLong((long) id);
}

static PyObject*
mxArray_mxGetClassName(PyObject* self)
{
  const char* class = mxGetClassName(mxArrayPtr(self));
  return PyString_FromString(class);
}

/* TODO: Add keyword option to index from 1 instead of 0 */
static PyObject*
mxArray_mxCalcSingleSubscript(PyObject* self, PyObject* args)
{
  mxArray* mxobj = mxArrayPtr(self);
  mwSize len = (mwIndex) PySequence_Length(args);
  mwSize dims = mxGetNumberOfDimensions(mxobj);
  if (len > dims) {
    return PyErr_Format(PyExc_IndexError, "Can't calculated %ld-dimensional subscripts for %ld-dimensional array",
			(long) len, (long) dims);
  }
  mwIndex subs[len];
  mwIndex i;
  for (i=0; i<len; i++)
    subs[i] = (mwIndex) PyInt_AsLong(PyTuple_GetItem(args, (Py_ssize_t) i));
  return PyLong_FromLong(mxCalcSingleSubscript(mxobj, len, subs));
}

/* TODO: Allow multiple subscript indexing */
static PyObject*
mxArray_mxGetField(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static char* kwlist[] = {"fieldname","index", NULL};
  const mxArray* ptr = mxArrayPtr(self);
  if (!mxIsStruct(ptr))
    return PyErr_Format(PyExc_TypeError, "Expected struct, got %s", mxGetClassName(ptr));
  char* fieldname;
  long index = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|l", 
				   kwlist, &fieldname, &index))
    return NULL;
  const mwSize numel = mxGetNumberOfElements(ptr);
  if (index >= numel || index < 0)
    return PyErr_Format(PyExc_IndexError, "Index %ld out of bounds (0 <= i < %ld)", index, (long) numel);
  if (mxGetFieldNumber(mxArrayPtr(self), fieldname) < 0)
    return PyErr_Format(PyExc_KeyError, "Struct has no '%s' field.", fieldname);
  mxArray* item = mxGetField(ptr, (mwIndex) index, fieldname);
  return Any_mxArray_to_PyObject(item);
}

void test_handler(int signum) 
{
  mexPrintf("got signal %d\n", signum);
}

/* FIXME: mxGetProperty and mxSetProperty cause SIGABRT if there's some sort of error,
   but there doesn't seem to be any way to catch the signal because the MATLAB interpreter
   either changes my signal handler or there's a different handler for that thread or something. 
   This happens if you try to access a nonexistent property (there is no C-API function to determine
   valid properties), if you try to access a property but have insufficient rights (also no way of
   checking that from C), or if an error occurs in the property's accessor function.
   Should probably use subsref/subsasgn instead. 
*/
static PyObject*
mxArray_mxGetProperty(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static char* kwlist[] = {"propname","index", NULL};
  const mxArray* ptr = mxArrayPtr(self);
  char* propname;
  long index = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|l", 
				   kwlist, &propname, &index))
    return NULL;
  const mwSize numel = mxGetNumberOfElements(ptr);
  if (index >= numel || index < 0)
    return PyErr_Format(PyExc_IndexError, "Index %ld out of bounds (0 <= i < %ld)", index, (long) numel);
  mxArray* item = mxGetProperty(ptr, (mwIndex) index, propname);
  if (!item)
    return PyErr_Format(PyExc_KeyError, "Lookup of property '%s' failed.", propname);
  return Any_mxArray_to_PyObject(item);
}

static PyObject*
mxArray_mxGetCell(PyObject* self, PyObject* args)
{
  long index = 0;
  const mxArray* ptr = mxArrayPtr(self);
  if (!mxIsCell(ptr))
    return PyErr_Format(PyExc_TypeError, "Expected cell, got %s", mxGetClassName(ptr));
  if (!PyArg_ParseTuple(args, "l", &index))
    return NULL;
  const mwSize numel = mxGetNumberOfElements(ptr);
  if (index >= numel || index < 0)
    return PyErr_Format(PyExc_IndexError, "Index %ld out of bounds (0 <= i < %ld)", index, (long) numel);
  mxArray* item = mxGetCell(ptr, (mwIndex) index);
  return Any_mxArray_to_PyObject(item);
}

/* helper function, initializes a new field of a struct. */
static int mxAddField_AndInit(mxArray* obj, const char* fieldname, mxArray* fillval) {
  if (!fillval) fillval = mxCreateDoubleMatrix(0,0,mxREAL);
  mwSize len = mxGetNumberOfElements(obj);
  mwIndex i;
  int fieldnum = mxAddField(obj, fieldname);
  if (fieldnum < 0) return fieldnum;
  for (i=0; i<len; i++) {
    mxArray* nextval = mxDuplicateArray(fillval);
    mexMakeArrayPersistent(nextval); /* FIXME: Use this line only if linking with libmex */
    mxSetFieldByNumber(obj, i, fieldnum, nextval);
  }
  return fieldnum;
}

/* TODO: Struct resizing */
/* TODO: Allow multiple subscript indexing */
static PyObject*
mxArray_mxSetField(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static char* kwlist[] = {"fieldname", "value", "index", NULL};
  const mxArray* ptr = mxArrayPtr(self);
  if (!mxIsStruct(ptr))
    return PyErr_Format(PyExc_TypeError, "Expected struct, got %s", mxGetClassName(ptr));
  char* fieldname;
  PyObject* newvalue;
  long index = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO|l",
				   kwlist, &fieldname, &newvalue, &index))
    return NULL;
  const mwSize numel = mxGetNumberOfElements(ptr);
  if (index >= numel || index < 0)
    return PyErr_Format(PyExc_IndexError, "Index %ld out of bounds (0 <= i < %ld)", index, (long) numel);
  if (mxGetFieldNumber(ptr, fieldname) < 0)
    if (mxAddField_AndInit((mxArray*) ptr, fieldname, NULL) < 0)
      return PyErr_Format(PyExc_KeyError, "Struct has no '%s' field, and could not create it.", fieldname);
  mxArray* mxvalue = mxDuplicateArray(Any_PyObject_to_mxArray(newvalue)); /*FIXME: This probably leaks when the input isn't already an mxArray, since the returned object is new but never freed */
  mexMakeArrayPersistent(mxvalue);
  mxArray* oldval = mxGetField(ptr, (mwIndex) index, fieldname);
  if (oldval) mxDestroyArray(oldval);
  mxSetField((mxArray*) ptr, (mwIndex) index, fieldname, mxvalue);
  Py_RETURN_NONE;
}

/* FIXME: See discussion at mxGetProperty */
static PyObject*
mxArray_mxSetProperty(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static char* kwlist[] = {"propname", "value", "index", NULL};
  const mxArray* ptr = mxArrayPtr(self);
  char* propname;
  PyObject* newvalue;
  long index = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO|l",
				   kwlist, &propname, &newvalue, &index))
    return NULL;
  const mwSize numel = mxGetNumberOfElements(ptr);
  if (index >= numel || index < 0)
    return PyErr_Format(PyExc_IndexError, "Index %ld out of bounds (0 <= i < %ld)", index, (long) numel);
  mxArray* mxvalue = Any_PyObject_to_mxArray(newvalue);
  mxSetProperty((mxArray*) ptr, (mwIndex) index, propname, mxvalue);
  Py_RETURN_NONE;
}

static PyObject*
mxArray_mxSetCell(PyObject* self, PyObject* args)
{
  const mxArray* ptr = mxArrayPtr(self);
  if (!mxIsCell(ptr))
    return PyErr_Format(PyExc_TypeError, "Expected cell, got %s", mxGetClassName(ptr));
  PyObject* newvalue;
  long index = 0;
  if (!PyArg_ParseTuple(args, "lO", &index, &newvalue))
    return NULL;
  const mwSize numel = mxGetNumberOfElements(ptr);
  if (index >= numel || index < 0)
    return PyErr_Format(PyExc_IndexError, "Index %ld out of bounds (0 <= i < %ld)", index, (long) numel);
  mxArray* mxvalue = mxDuplicateArray(Any_PyObject_to_mxArray(newvalue));
  mexMakeArrayPersistent(mxvalue);
  mxArray* oldval = mxGetCell(ptr, (mwIndex) index);
  if (oldval) mxDestroyArray(oldval);
  mxSetCell((mxArray*) ptr, (mwIndex) index, mxvalue);
  Py_RETURN_NONE;
}

static PyObject*
mxArray_mxGetFields(PyObject* self)
{
  mxArray* ptr = mxArrayPtr(self);
  int nfields = mxGetNumberOfFields(ptr);
  PyObject* outtuple = PyTuple_New(nfields);
  int i;
  for (i=0; i<nfields; i++) {
    const char* fieldname = mxGetFieldNameByNumber(ptr, i);
    if (!fieldname)
      return PyErr_Format(PyExc_RuntimeError, "Unable to read field %d from struct.", i);
    PyObject* pyname = PyString_FromString(fieldname);
    PyTuple_SetItem(outtuple, i, pyname);
  }
  return outtuple;
}

static PyObject*
mxArray_mxGetNumberOfElements(PyObject* self)
{
  mwSize len = mxGetNumberOfElements(mxArrayPtr(self));
  return PyLong_FromLong(len);
}

static PyObject*
mxArray_mxGetNumberOfDimensions(PyObject* self)
{
  mwSize ndims = mxGetNumberOfDimensions(mxArrayPtr(self));
  return PyLong_FromLong(ndims);
}

static PyObject*
mxArray_mxGetDimensions(PyObject* self)
{
  mwSize ndims = mxGetNumberOfDimensions(mxArrayPtr(self));
  PyObject* dimtuple = PyTuple_New(ndims);
  const mwSize* dimarray = mxGetDimensions(mxArrayPtr(self));
  Py_ssize_t i;
  for (i=0; i<ndims; i++) {
    PyTuple_SetItem(dimtuple, i, PyInt_FromSsize_t((Py_ssize_t) dimarray[i]));
  }
  return dimtuple;
}

static PyObject*
mxArray_mxGetElementSize(PyObject* self)
{
  size_t sz = mxGetElementSize(mxArrayPtr(self));
  return PyLong_FromSize_t(sz);
}

static PyMethodDef mxArray_methods[] = {
  {"mxGetClassID", (PyCFunction)mxArray_mxGetClassID, METH_NOARGS,
   "Returns the ClassId of the mxArray"},
  {"mxGetClassName", (PyCFunction)mxArray_mxGetClassName, METH_NOARGS,
   "Returns the name of the class of the mxArray"},
  {"mxCalcSingleSubscript", (PyCFunction)mxArray_mxCalcSingleSubscript, METH_VARARGS,
   "Calculates the linear index for the given subscripts"},
  {"mxGetField", (PyCFunction)mxArray_mxGetField, METH_VARARGS | METH_KEYWORDS,
   "Retrieve a field of a struct, optionally at a particular index."},
  {"mxGetProperty", (PyCFunction)mxArray_mxGetProperty, METH_VARARGS | METH_KEYWORDS,
   "Retrieve a property of a object, optionally at a particular index."},
  {"mxGetCell", (PyCFunction)mxArray_mxGetCell, METH_VARARGS,
   "Retrieve a cell array element"},
  {"mxSetField", (PyCFunction)mxArray_mxSetField, METH_VARARGS | METH_KEYWORDS,
   "Set a field of a struct, optionally at a particular index."},
  {"mxSetProperty", (PyCFunction)mxArray_mxSetProperty, METH_VARARGS | METH_KEYWORDS,
   "Set a property of an object, optionally at a particular index."},
  {"mxSetCell", (PyCFunction)mxArray_mxSetCell, METH_VARARGS,
   "Set a cell array element"},
  {"mxGetFields", (PyCFunction)mxArray_mxGetFields, METH_NOARGS,
   "Returns a tuple with the field names of the struct."},
  {"mxGetNumberOfElements", (PyCFunction)mxArray_mxGetNumberOfElements, METH_NOARGS,
   "Returns the number of elements in the array."},
  {"mxGetNumberOfDimensions", (PyCFunction)mxArray_mxGetNumberOfDimensions, METH_NOARGS,
   "Returns the number of dimensions of the array."},
  {"mxGetDimensions", (PyCFunction)mxArray_mxGetDimensions, METH_NOARGS,
   "Returns a tuple containing the sizes of each dimension"},
  {"mxGetElementSize", (PyCFunction)mxArray_mxGetElementSize, METH_NOARGS,
   "Returns the size of each element in the array, in bytes."},
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
