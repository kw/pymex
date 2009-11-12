#include <Python.h>
#include "structmember.h"
#if PYMEX_USE_NUMPY
 #define PY_ARRAY_UNIQUE_SYMBOL PYMEX_ARRAY_API
 #define NPY_USE_PYMEM 1
 #include <numpy/arrayobject.h>
#endif
#define LIBMEXMODULE
#include "pymex.h"
#include <mex.h>
#include <signal.h>

#define PY3K_VERSION_HEX 0x3000000

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
  return PyLong_FromLong((long) id);
}

static PyObject*
mxArray_mxGetClassName(PyObject* self)
{
  const char* class = mxGetClassName(mxArrayPtr(self));
  return PyBytes_FromString(class);
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
    subs[i] = (mwIndex) PyLong_AsLong(PyTuple_GetItem(args, (Py_ssize_t) i));
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
    PyObject* pyname = PyBytes_FromString(fieldname);
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
    PyTuple_SetItem(dimtuple, i, PyLong_FromSsize_t((Py_ssize_t) dimarray[i]));
  }
  return dimtuple;
}

static PyObject*
mxArray_mxGetElementSize(PyObject* self)
{
  size_t sz = mxGetElementSize(mxArrayPtr(self));
  return PyLong_FromSize_t(sz);
}

static PyObject* 
mxArray_float(PyObject* self)
{
  mxArray* ptr = mxArrayPtr(self);
  if (ptr && !mxIsEmpty(ptr) && (mxIsNumeric(ptr) || mxIsLogical(ptr) || mxIsChar(ptr))) {
    double val = mxGetScalar(ptr);
    return PyFloat_FromDouble(val);
  }
  else {
    if (!ptr)
      return PyErr_Format(PyExc_ValueError, "Null pointer");
    else if (mxIsEmpty(ptr))
      return PyErr_Format(PyExc_ValueError, "Empty array");
    else
      return PyErr_Format(PyExc_ValueError, "Expected numeric, logical, or character array, got %s instead.", mxGetClassName(ptr));
  }
}

static PyObject* 
mxArray_long(PyObject* self)
{
  mxArray* ptr = mxArrayPtr(self);
  if (ptr && !mxIsEmpty(ptr) && (mxIsNumeric(ptr) || mxIsLogical(ptr) || mxIsChar(ptr))) {
    long long val = (long long) mxGetScalar(ptr);
    return PyLong_FromLongLong(val);
  }
  else {
    if (!ptr)
      return PyErr_Format(PyExc_ValueError, "Null pointer");
    else if (mxIsEmpty(ptr))
      return PyErr_Format(PyExc_ValueError, "Empty array");
    else
      return PyErr_Format(PyExc_ValueError, "Expected numeric, logical, or character array, got %s instead.", mxGetClassName(ptr));
  }
}

#if PY_VERSION_HEX < PY3K_VERSION_HEX
static PyObject* 
mxArray_int(PyObject* self)
{
  mxArray* ptr = mxArrayPtr(self);
  if (ptr && !mxIsEmpty(ptr) && (mxIsNumeric(ptr) || mxIsLogical(ptr) || mxIsChar(ptr))) {
    long val = (long) mxGetScalar(ptr);
    return PyLong_FromLong(val);
  }
  else {
    if (!ptr)
      return PyErr_Format(PyExc_ValueError, "Null pointer");
    else if (mxIsEmpty(ptr))
      return PyErr_Format(PyExc_ValueError, "Empty array");
    else
      return PyErr_Format(PyExc_ValueError, "Expected numeric, logical, or character array, got %s instead.", mxGetClassName(ptr));
  }
}
#endif

static PyObject*
mxArray_index(PyObject* self)
{
  return mxArray_long(self);
}


/* These two are helper methods. The stringifier has to talk to MATLAB, so it might not be available. Because
   of this, str and repr must try their preferences and fall back on the simple repr if necessary. */
static PyObject* mxArray_str_helper(PyObject* self) {
  PyObject* rawstring = PyObject_CallMethod(libmexmodule, "call", "sO", "any2str", self);
  PyObject* stringval = PyObject_CallMethod(rawstring, "strip", "");
  Py_DECREF(rawstring);
  return stringval;
}
static PyObject* mxArray_repr_helper(PyObject* self) {
  mxArray* ptr = mxArrayPtr(self);
  return PyBytes_FromFormat("<%s at %p>", mxGetClassName(ptr), ptr);
}

static PyObject*
mxArray_str(PyObject* self)
{
  PyObject* str = mxArray_str_helper(self);
  if (PyErr_Occurred()) {
    PyErr_Clear();
    return mxArray_repr_helper(self);
  }
  else return str;
}

static PyObject*
mxArray_repr(PyObject* self)
{
  mxArray* ptr = mxArrayPtr(self);
  if ((mxIsNumeric(ptr) || mxIsLogical(ptr) || mxIsChar(ptr)) && mxGetNumberOfElements(ptr) < 16) {
    return mxArray_str(self);
  }
  else return mxArray_repr_helper(self);     
}

/* This definition shamelessly copied from NumPy to remove dependence on it for building. */
#if !PYMEX_USE_NUMPY
typedef struct {
  int two;              /* contains the integer 2 -- simple sanity check */
  int nd;               /* number of dimensions */
  char typekind;        /* kind in array --- character code of typestr */
  int itemsize;         /* size of each element */
  int flags;            /* flags indicating how the data should be interpreted */
                        /*   must set ARR_HAS_DESCR bit to validate descr */
  Py_intptr_t *shape;   /* A length-nd array of shape information */
  Py_intptr_t *strides; /* A length-nd array of stride information */
  void *data;           /* A pointer to the first element of the array */
  PyObject *descr;      /* NULL or data-description (same as descr key
                                of __array_interface__) -- must set ARR_HAS_DESCR
                                flag or this will be ignored. */
} PyArrayInterface;
#define NPY_CONTIGUOUS    0x0001
#define NPY_FORTRAN       0x0002
#define NPY_ALIGNED       0x0100
#define NPY_NOTSWAPPED    0x0200
#define NPY_WRITEABLE     0x0400
#define NPY_ARR_HAS_DESCR  0x0800
#endif

static void numpy_array_struct_destructor(void* ptr, void* desc)
{
  PyMem_Free(ptr);
  Py_DECREF(desc);
}

static PyObject* mxArray_numpy_array_struct(PyObject* self, void* closure)
{
  PyArrayInterface* info = PyMem_New(PyArrayInterface, 1);
  mxArray* ptr = mxArrayPtr(self);
  info->two = 2;
  info->nd = (int) mxGetNumberOfDimensions(ptr);
  info->typekind = mxClassID_to_Numpy_Typekind(mxGetClassID(ptr));
  info->itemsize = (int) mxGetElementSize(ptr);
  info->flags = NPY_FORTRAN | NPY_ALIGNED | NPY_NOTSWAPPED | NPY_WRITEABLE;
  info->shape = PyMem_New(Py_intptr_t, info->nd);
  int i;
  const mwSize* dims = mxGetDimensions(ptr);
  for(i=0; i<info->nd; i++)
    info->shape[i] = (Py_intptr_t) dims[i];
  info->strides = NULL;
  info->data = mxGetData(ptr);
  info->descr = NULL;
  Py_INCREF(self);  
  return PyCObject_FromVoidPtrAndDesc(info, self, numpy_array_struct_destructor);
}

static PyGetSetDef mxArray_getseters[] = {
    {"__array_struct__", 
     (getter)mxArray_numpy_array_struct, NULL, 
     "NumPy array interface",
     NULL},
    {NULL}  /* Sentinel */
};


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

#if PY_VERSION_HEX < PY3K_VERSION_HEX
static PyNumberMethods mxArray_numbermethods = {
  0, /*binaryfunc nb_add;*/
  0, /*binaryfunc nb_subtract;*/
  0, /*binaryfunc nb_multiply;*/
  0, /*binaryfunc nb_divide;*/
  0, /*binaryfunc nb_remainder;*/
  0, /*binaryfunc nb_divmod;*/
  0, /*ternaryfunc nb_power;*/
  0, /*unaryfunc nb_negative;*/
  0, /*unaryfunc nb_positive;*/
  0, /*unaryfunc nb_absolute;*/
  0, /*inquiry nb_nonzero;     */
  0, /*unaryfunc nb_invert;*/
  0, /*binaryfunc nb_lshift;*/
  0, /*binaryfunc nb_rshift;*/
  0, /*binaryfunc nb_and;*/
  0, /*binaryfunc nb_xor;*/
  0, /*binaryfunc nb_or;*/
  0, /*coercion nb_coerce;     */
  mxArray_int, /*unaryfunc nb_int;*/
  mxArray_long, /*unaryfunc nb_long;*/
  mxArray_float, /*unaryfunc nb_float;*/
  0, /*unaryfunc nb_oct;*/
  0, /*unaryfunc nb_hex;*/
  0, /*binaryfunc nb_inplace_add;*/
  0, /*binaryfunc nb_inplace_subtract;*/
  0, /*binaryfunc nb_inplace_multiply;*/
  0, /*binaryfunc nb_inplace_divide;*/
  0, /*binaryfunc nb_inplace_remainder;*/
  0, /*ternaryfunc nb_inplace_power;*/
  0, /*binaryfunc nb_inplace_lshift;*/
  0, /*binaryfunc nb_inplace_rshift;*/
  0, /*binaryfunc nb_inplace_and;*/
  0, /*binaryfunc nb_inplace_xor;*/
  0, /*binaryfunc nb_inplace_or;*/
  0, /*binaryfunc nb_floor_divide;*/
  0, /*binaryfunc nb_true_divide;*/
  0, /*binaryfunc nb_inplace_floor_divide;*/
  0, /*binaryfunc nb_inplace_true_divide;*/
  mxArray_index, /*unaryfunc nb_index;*/
};
#else /* Py3k */
static PyNumberMethods mxArray_numbermethods = {
  0, /* binaryfunc nb_add */
  0, /* binaryfunc nb_subtract */
  0, /* binaryfunc nb_multiply */
  0, /* binaryfunc nb_remainder */
  0, /* binaryfunc nb_divmod */
  0, /* ternaryfunc nb_power */
  0, /* unaryfunc nb_negative */
  0, /* unaryfunc nb_positive */
  0, /* unaryfunc nb_absolute */
  0, /* inquiry nb_bool */
  0, /* unaryfunc nb_invert */
  0, /* binaryfunc nb_lshift */
  0, /* binaryfunc nb_rshift */
  0, /* binaryfunc nb_and */
  0, /* binaryfunc nb_xor */
  0, /* binaryfunc nb_or */
  mxArray_long, /* unaryfunc nb_int */
  0, /* void *nb_reserved */
  mxArray_float, /* unaryfunc nb_float */
  0, /* binaryfunc nb_inplace_add */
  0, /* binaryfunc nb_inplace_subtract */
  0, /* binaryfunc nb_inplace_multiply */
  0, /* binaryfunc nb_inplace_remainder */
  0, /* ternaryfunc nb_inplace_power */
  0, /* binaryfunc nb_inplace_lshift */
  0, /* binaryfunc nb_inplace_rshift */
  0, /* binaryfunc nb_inplace_and */
  0, /* binaryfunc nb_inplace_xor */
  0, /* binaryfunc nb_inplace_or */
  0, /* binaryfunc nb_floor_divide */
  0, /* binaryfunc nb_true_divide */
  0, /* binaryfunc nb_inplace_floor_divide */
  0, /* binaryfunc nb_inplace_true_divide */
  mxArray_index, /* unaryfunc nb_index */
};

#endif

#if PY_VERSION_HEX < PY3K_VERSION_HEX
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
    mxArray_repr,                         /*tp_repr*/
    &mxArray_numbermethods,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    mxArray_str,                         /*tp_str*/
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
    mxArray_getseters,        /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    0,                        /* tp_new */
};
#else /* Py3k */
static PyTypeObject mxArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "libmex.mxArray",             /* tp_name */
    sizeof(mxArrayObject),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)mxArray_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    mxArray_repr,                         /* tp_repr */
    &mxArray_numbermethods,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    mxArray_str,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "mxArray objects",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    mxArray_methods,             /* tp_methods */
    0,                        /* tp_members */
    mxArray_getseters,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};
#endif

#if PY_VERSION_HEX >= PY3K_VERSION_HEX
static PyModuleDef libmexmodule_def = {
    PyModuleDef_HEAD_INIT,
    "libmex",
    "Embedded mex API module",
    -1,
    libmex_methods, NULL, NULL, NULL, NULL
};
#endif

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC PyObject*
#endif
PyMODINIT_FUNC
initlibmexmodule(void)
{
  mxArrayType.tp_new = PyType_GenericNew;
  /*mxArrayType.tp_as_number = mxArray_numbermethods;*/

  #if PY_VERSION_HEX < PY3K_VERSION_HEX
  if (PyType_Ready(&mxArrayType) < 0) return;
  libmexmodule = Py_InitModule("libmex", libmex_methods);
  if (!libmexmodule) return;
  #else
  if (PyType_Ready(&mxArrayType) < 0) return NULL;
  libmexmodule = PyModule_Create(&libmexmodule_def);
  if (!libmexmodule) return NULL;
  #endif


  Py_INCREF(&mxArrayType);
  PyModule_AddObject(libmexmodule, "mxArray", (PyObject*) &mxArrayType);

  #if PYMEX_USE_NUMPY
  /* numpy C-API import */
  import_array();
  #endif

  #if PY_VERSION_HEX >= PY3K_VERSION_HEX
  return libmexmodule;
  #endif
}
