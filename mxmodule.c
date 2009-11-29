/* Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
   For full license details, see the LICENSE file. */

#define MXMODULE
#import "pymex.h"
#import "structmember.h"

static PyObject* dowrap(PyObject* cobj) {
  PyObject* nargs = PyTuple_New(0);
  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "mxpointer", cobj);
  PyObject* arraycls = Find_mltype_for(mxArrayPtr(cobj));
  /* TODO: There is probably a better way to do this... */
  PyObject* ret = PyObject_Call(arraycls, nargs, kwargs);
  Py_DECREF(nargs);
  Py_DECREF(kwargs);
  Py_DECREF(arraycls);
  return ret;
}


#define DIMS_FROM_SEQ(A)						\
  mwSize ndim = PySequence_Size(A);					\
  mwSize dims[ndim];							\
  mwSize i;								\
  for (i=0; i < ndim; i++) {						\
    PyObject* item = PySequence_GetItem(A, i);				\
    if (!item) return NULL;						\
    PyObject* index = PyNumber_Index(item);				\
    Py_DECREF(item);							\
    if (!index) return NULL;						\
    dims[i] = (mwSize) PyLong_AsLong(index);				\
    Py_DECREF(index);							\
    if (PyErr_Occurred()) return NULL;					\
    if (dims[i] < 0)							\
      return PyErr_Format(PyExc_ValueError,				\
			  "dims[%ld] = %ld, should be non-negative",	\
			  (long) i, (long) dims[i]);			\
  } if(1)
  
static PyObject*
CreateCellArray(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static char* kwlist[] = {"dims", "wrap", NULL};
  PyObject* pydims = NULL;
  int wrap = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oi", kwlist,
				   &pydims, &wrap))
    return NULL;
  if (!pydims) pydims = PyTuple_New(0);
  else Py_INCREF(pydims);
  DIMS_FROM_SEQ(pydims);
  Py_DECREF(pydims);
  mxArray* cell = mxCreateCellArray(ndim, dims);
  if (wrap)    
    return dowrap(mxArrayPtr_New(cell));
  else
    return mxArrayPtr_New(cell);
}

static PyObject*
CreateNumericArray(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static char* kwlist[] = {"dims", "mxclass", "complexity", "wrap", NULL};
  PyObject* pydims = NULL;
  mxClassID class = mxDOUBLE_CLASS;
  mxComplexity complexity = mxREAL;
  int wrap = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oiii", kwlist, 
				   &pydims, &class, &complexity, &wrap))
    return NULL;
  if (!pydims) pydims = PyTuple_New(0);
  else Py_INCREF(pydims);
  DIMS_FROM_SEQ(pydims);
  Py_DECREF(pydims);
  mxArray* array = mxCreateNumericArray(ndim, dims, class, complexity);
  if (wrap)
    return dowrap(mxArrayPtr_New(array));
  else
    return mxArrayPtr_New(array);
}

/* TODO: Allow field initialization. */
static PyObject*
CreateStructArray(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static char* kwlist[] = {"dims", "wrap", NULL};
  PyObject* pydims = NULL;
  int wrap = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oi", kwlist,
				   &pydims, &wrap))
    return NULL;
  if (!pydims) pydims = PyTuple_Pack(1, PyLong_FromLong(1));
  else Py_INCREF(pydims);
  DIMS_FROM_SEQ(pydims);
  Py_DECREF(pydims);
  mxArray* array = mxCreateStructArray(ndim, dims, 0, NULL);
  if (wrap)
    return dowrap(mxArrayPtr_New(array));
  else
    return mxArrayPtr_New(array);
}

static PyObject*
CreateCharArray(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static char* kwlist[] = {"dims", "wrap", NULL};
  PyObject* pydims = NULL;
  int wrap = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oi", kwlist,
				   &pydims, &wrap))
    return NULL;
  if (!pydims) pydims = PyTuple_New(0);
  else Py_INCREF(pydims);
  DIMS_FROM_SEQ(pydims);
  Py_DECREF(pydims);
  mxArray* array = mxCreateCharArray(ndim, dims);
  if (wrap)
    return dowrap(mxArrayPtr_New(array));
  else
    return mxArrayPtr_New(array);
}

static PyObject*
CreateString(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static char* kwlist[] = {"string", "wrap", NULL};
  char* string = NULL;
  int wrap = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|i", kwlist,
				   &string, &wrap))
    return NULL;
  mxArray* array = mxCreateString((const char*) string);
  if (wrap)
    return dowrap(mxArrayPtr_New(array));
  else
    return mxArrayPtr_New(array);
}

static PyObject*
CreateFunctionHandle(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static char* kwlist[] = {"name", "closure", "wrap", NULL};
  char* name = NULL;
  char* closure = NULL;
  int wrap = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ssi", kwlist,
				   &name, &closure, &wrap))
    return NULL;
  PyObject* evalstring;
  if (name && closure)
    return PyErr_Format(PyExc_ValueError, "Provide a function name OR a closure, not both.");
  else if (name)
    evalstring = PyBytes_FromFormat("@%s;", name);
  else if (closure)
    evalstring = PyBytes_FromFormat("%s;", closure);
  else
    return PyErr_Format(PyExc_ValueError, "Must provide a string containing a name or a MATLAB closure.");
  #if MATLAB_MEX_FILE
  PyObject* result = PyObject_CallMethod(mexmodule, "eval", "O", evalstring);
  Py_DECREF(evalstring);
  if (!result) return NULL;
  if (mxGetClassID(mxArrayPtr(result)) != mxFUNCTION_CLASS)
    return PyErr_Format(PyExc_ValueError, "Returned value was not a function handle?");  
  if (wrap)
    return result;
  else {
    PyObject* cobj = PyObject_GetAttrString(result, "_mxptr");
    Py_DECREF(result);    
    return cobj;
  }
  #else
  Py_DECREF(evalstring);
  return PyErr_String(MATLABError, "Function handles can't presently "
		      "be instantiated outside of MATLAB.");
  #endif
}

static PyObject*
wrap_pycobject(PyObject* self, PyObject* args)
{
  PyObject* cobj = NULL;
  if (!PyArg_ParseTuple(args, "O", &cobj))
    return NULL;
  return dowrap(cobj);
}

static PyMethodDef mx_methods[] = {
  {"create_cell_array", (PyCFunction)CreateCellArray, METH_VARARGS | METH_KEYWORDS,
   "Creates a cell array with given dimensions, i.e., mx.create_cell_array(2,5,1). "
   "Initially populated by empty matrices."},
  {"create_numeric_array", (PyCFunction)CreateNumericArray, METH_VARARGS | METH_KEYWORDS,
   "Creates a numeric array with given tuple of dimensions. Specify class and complexity "
   "with one of the enum constants (defaults are mx.mxDOUBLE_CLASS and mx.mxREAL, respectively). "
   "Example: mx.create_numeric_array((5,4,2), mx.mxDOUBLE_CLASS, mx.mxREAL)"},
  {"create_struct_array", (PyCFunction)CreateStructArray, METH_VARARGS | METH_KEYWORDS, 
   "Creates a struct array with no fields."},
  {"create_char_array", (PyCFunction)CreateCharArray, METH_VARARGS | METH_KEYWORDS,
   "Creates a character array with the given dimensions."},
  {"create_string", (PyCFunction)CreateString, METH_VARARGS | METH_KEYWORDS,
   "Creates a character array from the given string."},
  {"create_function_handle", (PyCFunction)CreateFunctionHandle, METH_VARARGS | METH_KEYWORDS,
   "If called with name='somefunc', returns a handle to that function. "
   "If called with closure='@(x) x+1', returns a MATLAB lambda function. "
   "Note that closures created this way are closed over the current base workspace."},
  {"wrap_pycobject", (PyCFunction)wrap_pycobject, METH_VARARGS,
   "Locates an appropriate class for the given mxArray pointer and returns the "
   "resulting instance."},
  {NULL, NULL, 0, NULL}
};

/* libmx mxArray type */

static int
mxArray_init(mxArrayObject* self, PyObject* args, PyObject* kwargs)
{
  PyObject* mxptr = PyDict_GetItemString(kwargs, "mxpointer");
  if (!mxptr) {
    PyErr_Format(PyExc_ValueError, "Failed to build mx.Array wrapper: need mxpointer keyword argument.");
    return -1;
  }
  if (!mxArrayPtr_Check(mxptr)) {
    PyErr_Format(PyExc_TypeError, "mxpointer must be a valid mxArrayPtr");
    return -1;
  }
  Py_INCREF(mxptr);
  self->mxptr = mxptr;
  return 0;
}

static void 
mxArray_dealloc(mxArrayObject* self)
{
  Py_XDECREF(self->mxptr);
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
    return PyErr_Format(PyExc_IndexError, "Can't calculate %ld-dimensional subscripts for %ld-dimensional array",
			(long) len, (long) dims);
  }
  mwIndex subs[len];
  mwIndex i;
  for (i=0; i<len; i++) {
    PyObject* ind = PyNumber_Index(PyTuple_GetItem(args, (Py_ssize_t) i));
    if (PyErr_Occurred()) return NULL;
    subs[i] = (mwIndex) PyLong_AsLong(ind);
  }
  return PyLong_FromLong(mxCalcSingleSubscript(mxobj, len, subs));
}

static PyObject*
mxArray_mxGetField(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static char* kwlist[] = {"fieldname","index", NULL};
  mxArray* ptr = mxArrayPtr(self);
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
  if (!item) {
    item = mxCreateDoubleMatrix(0,0,mxREAL);
    PERSIST_ARRAY(item);
    mxSetField(ptr, (mwIndex) index, fieldname, item);
    item = mxGetField(ptr, (mwIndex) index, fieldname);
  }
  return Any_mxArray_to_PyObject(item);
}

/* FIXME: mxGetProperty and mxSetProperty cause SIGABRT if there's some sort of error,
   but there doesn't seem to be any way to catch the signal because the MATLAB interpreter
   either changes my signal handler or there's a different handler for that thread or something. 
   This happens if you try to access a nonexistent property (there is no C-API function to determine
   valid properties), if you try to access a property but have insufficient rights (also no way of
   checking that from C), or if an error occurs in the property's accessor function.
   Should probably use subsref/subsasgn instead. 

   UPDATE: Apparently this is fixed in 2009b, but with some change to the API of some sort.
   I do not have this version. I did notice that on my 2009a machine no SIGABRT was signaled. Odd.
*/
static PyObject*
mxArray_mxGetProperty(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static char* kwlist[] = {"propname","index", NULL};
  mxArray* ptr = mxArrayPtr(self);
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
  mxArray* ptr = mxArrayPtr(self);
  if (!mxIsCell(ptr))
    return PyErr_Format(PyExc_TypeError, "Expected cell, got %s", mxGetClassName(ptr));
  if (!PyArg_ParseTuple(args, "l", &index))
    return NULL;
  const mwSize numel = mxGetNumberOfElements(ptr);
  if (index >= numel || index < 0)
    return PyErr_Format(PyExc_IndexError, "Index %ld out of bounds (0 <= i < %ld)", index, (long) numel);
  mxArray* item = mxGetCell(ptr, (mwIndex) index);
  if (!item) {
    item = mxCreateDoubleMatrix(0,0,mxREAL);
    PERSIST_ARRAY(item);
    mxSetCell(ptr, (mwIndex) index, item);
    item = mxGetCell(ptr, (mwIndex) index);
  }
  return Any_mxArray_to_PyObject(item);
}

static PyObject*
mxArray_mxSetField(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static char* kwlist[] = {"fieldname", "value", "index", NULL};
  mxArray* ptr = mxArrayPtr(self);
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
    if (mxAddField(ptr, fieldname) < 0)
      return PyErr_Format(PyExc_KeyError, "Struct has no '%s' field, and could not create it.", fieldname);
  mxArray* mxvalue = mxDuplicateArray(Any_PyObject_to_mxArray(newvalue)); /*FIXME: This probably leaks when the input isn't already an mxArray, since the returned object is new but never freed */
  PERSIST_ARRAY(mxvalue);
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
  PERSIST_ARRAY(mxvalue);
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
  mwSize ndim = mxGetNumberOfDimensions(mxArrayPtr(self));
  return PyLong_FromLong(ndim);
}

static PyObject*
mxArray_mxGetDimensions(PyObject* self)
{
  mwSize ndim = mxGetNumberOfDimensions(mxArrayPtr(self));
  PyObject* dimtuple = PyTuple_New(ndim);
  const mwSize* dimarray = mxGetDimensions(mxArrayPtr(self));
  Py_ssize_t i;
  for (i=0; i<ndim; i++) {
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
  if (ptr && mxGetNumberOfElements(ptr) == 1 && (mxIsNumeric(ptr) || mxIsLogical(ptr) || mxIsChar(ptr))) {
    double val = mxGetScalar(ptr);
    return PyFloat_FromDouble(val);
  }
  else {
    if (!ptr)
      return PyErr_Format(PyExc_ValueError, "Null pointer");
    else if (mxIsEmpty(ptr))
      return PyErr_Format(PyExc_ValueError, "Empty array");
    else if (mxGetNumberOfElements(ptr) > 1)
      return PyErr_Format(PyExc_ValueError, "Not a scalar");
    else
      return PyErr_Format(PyExc_ValueError, "Expected numeric, logical, or character array, got %s instead.", mxGetClassName(ptr));
  }
}

static PyObject* 
mxArray_long(PyObject* self)
{
  mxArray* ptr = mxArrayPtr(self);
  if (ptr && mxGetNumberOfElements(ptr) == 1 && (mxIsNumeric(ptr) || mxIsLogical(ptr) || mxIsChar(ptr))) {
    /* FIXME: This is a bad idea. Look up the correct type and interpret appropriately. */
    long long val = (long long) mxGetScalar(ptr);
    return PyLong_FromLongLong(val);
  }
  else {
    if (!ptr)
      return PyErr_Format(PyExc_ValueError, "Null pointer");
    else if (mxIsEmpty(ptr))
      return PyErr_Format(PyExc_ValueError, "Empty array");
    else if (mxGetNumberOfElements(ptr) > 1)
      return PyErr_Format(PyExc_ValueError, "Not a scalar");
    else
      return PyErr_Format(PyExc_ValueError, "Expected numeric, logical, or character array, got %s instead.", mxGetClassName(ptr));
  }
}

static PyObject*
mxArray_index(PyObject* self)
{
  return mxArray_long(self);
}


/* These two are helper methods. The stringifier has to talk to MATLAB, so it might not be available. Because
   of this, str and repr must try their preferences and fall back on the simple repr if necessary. */
static PyObject* mxArray_repr_helper(PyObject* self) {
  mxArray* ptr = mxArrayPtr(self);
  return PyBytes_FromFormat("<%s at %p>", mxGetClassName(ptr), ptr);
}

static PyObject* mxArray_str_helper(PyObject* self) {
  #if MATLAB_MEX_FILE
  static char newline[] = {10, 0};
  PyObject* rawstring = PyObject_CallMethod(mexmodule, "call", "sO", "any2str", self);
  PyObject* stringval = PyObject_CallMethod(rawstring, "strip", "s", newline);
  Py_DECREF(rawstring);
  return stringval;
  #else
  return mxArray_repr_helper(self);
  #endif
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
  #if MATLAB_MEX_FILE
  if ((mxIsNumeric(ptr) || mxIsLogical(ptr) || mxIsChar(ptr)) && mxGetNumberOfElements(ptr) < 16) {
    return mxArray_str(self);
  }
  #endif
  return mxArray_repr_helper(self);     
}

static void _dataptr_destructor(void* data, void* desc)
{
  Py_XDECREF((PyObject*) desc);
}
static PyObject*
mxArray_get_data(PyObject* self)
{
  mxArray* ptr = mxArrayPtr(self);
  void* data = mxGetData(ptr);
  Py_INCREF(self);
  PyObject* cobj = PyCObject_FromVoidPtrAndDesc(data, self, _dataptr_destructor);
  return cobj;
}

static PyObject*
mxArray_get_element(PyObject* self, PyObject* args)
{
  Py_ssize_t index = 0;
  if (!PyArg_ParseTuple(args, "|n", &index)) return NULL;
  mxArray* ptr = mxArrayPtr(self);
  Py_ssize_t numel = (Py_ssize_t) mxGetNumberOfElements(ptr);
  Py_ssize_t elsize = (Py_ssize_t) mxGetElementSize(ptr);
  if (index < 0 || index >= numel)
    PyErr_Format(PyExc_IndexError, "Array index out of bounds (0<=%zd<%zd)", index, numel);
  char* data = (char*) mxGetData(ptr);
  data += elsize*index;
  return PyBytes_FromStringAndSize(data, elsize);
}

static PyObject*
mxArray_set_element(PyObject* self, PyObject* args, PyObject* kw)
{
  static char* kwlist[] = {"bytes", "index", NULL};
  Py_ssize_t index = 0;
  PyObject* bytes = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kw, "S|n", kwlist, &bytes, &index)) return NULL;
  mxArray* ptr = mxArrayPtr(self);
  Py_ssize_t numel = (Py_ssize_t) mxGetNumberOfElements(ptr);
  Py_ssize_t elsize = (Py_ssize_t) mxGetElementSize(ptr);
  if (index < 0 || index >= numel)
    PyErr_Format(PyExc_IndexError, "Array index out of bounds (0<=%zd<%zd)", index, numel);
  Py_ssize_t bytesize = PyBytes_Size(bytes);
  if (bytesize != elsize)
    PyErr_Format(PyExc_ValueError, "Bytes size must equal element size (%zd != %zd)", 
		 bytesize, elsize);
  char* bytestring = PyBytes_AsString(bytes);
  char* data = (char*) mxGetData(ptr);
  data += elsize*index;
  int i;
  for (i=0; i < elsize; i++)
    data[i] = bytestring[i];
  Py_RETURN_NONE;
}

/* This definition shamelessly copied from NumPy to remove dependence on it for building. */
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
/* end things copied from NumPy */

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
  {"_get_class_id", (PyCFunction)mxArray_mxGetClassID, METH_NOARGS,
   "Returns the ClassId of the mxArray"},
  {"_get_class_name", (PyCFunction)mxArray_mxGetClassName, METH_NOARGS,
   "Returns the name of the class of the mxArray"},
  {"_calc_single_subscript", (PyCFunction)mxArray_mxCalcSingleSubscript, METH_VARARGS,
   "Calculates the linear index for the given subscripts"},
  {"_get_field", (PyCFunction)mxArray_mxGetField, METH_VARARGS | METH_KEYWORDS,
   "Retrieve a field of a struct, optionally at a particular index."},
  {"_get_property", (PyCFunction)mxArray_mxGetProperty, METH_VARARGS | METH_KEYWORDS,
   "Retrieve a property of a object, optionally at a particular index."},
  {"_get_cell", (PyCFunction)mxArray_mxGetCell, METH_VARARGS,
   "Retrieve a cell array element"},
  {"_set_field", (PyCFunction)mxArray_mxSetField, METH_VARARGS | METH_KEYWORDS,
   "Set a field of a struct, optionally at a particular index."},
  {"_set_property", (PyCFunction)mxArray_mxSetProperty, METH_VARARGS | METH_KEYWORDS,
   "Set a property of an object, optionally at a particular index."},
  {"_set_cell", (PyCFunction)mxArray_mxSetCell, METH_VARARGS,
   "Set a cell array element"},
  {"_get_fields", (PyCFunction)mxArray_mxGetFields, METH_NOARGS,
   "Returns a tuple with the field names of the struct."},
  {"_get_number_of_elements", (PyCFunction)mxArray_mxGetNumberOfElements, METH_NOARGS,
   "Returns the number of elements in the array."},
  {"_get_number_of_dimensions", (PyCFunction)mxArray_mxGetNumberOfDimensions, METH_NOARGS,
   "Returns the number of dimensions of the array."},
  {"_get_dimensions", (PyCFunction)mxArray_mxGetDimensions, METH_NOARGS,
   "Returns a tuple containing the sizes of each dimension"},
  {"_get_element_size", (PyCFunction)mxArray_mxGetElementSize, METH_NOARGS,
   "Returns the size of each element in the array, in bytes."},
  {"_get_data", (PyCFunction)mxArray_get_data, METH_NOARGS,
   "Returns a C object pointing to this array's data segment."},
  {"_get_element", (PyCFunction)mxArray_get_element, METH_VARARGS,
   "Gets a particular element from the array as a byte string. Default index=0"},
  {"_set_element", (PyCFunction)mxArray_set_element, METH_VARARGS | METH_KEYWORDS,
   "Given a string of bytes, sets the specified element to it. Default index=0"},
  {NULL}
};

static PyMemberDef mxArray_members[] = {
  {"_mxptr", T_OBJECT_EX, offsetof(mxArrayObject, mxptr), 0, 
   "CObject pointer to mxArray object"},
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
  mxArray_long, /*unaryfunc nb_int;*/
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
    "matlab.mx.Array",          /*tp_name*/
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
    mxArray_members,                        /* tp_members */
    mxArray_getseters,        /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)mxArray_init,                         /* tp_init */
    0,                         /* tp_alloc */
    0,                        /* tp_new */
};
#else /* Py3k */
static PyTypeObject mxArrayType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "matlab.mx.Array",             /* tp_name */
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
    mxArray_members,                        /* tp_members */
    mxArray_getseters,           /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)mxArray_init,                         /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};
#endif

#if PY_VERSION_HEX >= PY3K_VERSION_HEX
static PyModuleDef mxmodule_def = {
    PyModuleDef_HEAD_INIT,
    "matlab.mx",
    "MATLAB matrix API module",
    -1,
    mx_methods, NULL, NULL, NULL, NULL
};
#endif

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC PyObject*
#endif
PyMODINIT_FUNC
initmxmodule(void)
{
  mxArrayType.tp_new = PyType_GenericNew;
  /*mxArrayType.tp_as_number = mxArray_numbermethods;*/

  #if PY_VERSION_HEX < PY3K_VERSION_HEX
  if (PyType_Ready(&mxArrayType) < 0) return;
  PyObject* m = Py_InitModule3("matlab.mx", mx_methods, "MATLAB matrix API module");
  if (!m) return;
  #else
  if (PyType_Ready(&mxArrayType) < 0) return NULL;
  PyObject* m = PyModule_Create(&mxmodule_def);
  if (!m) return NULL;
  #endif

  Py_INCREF(&mxArrayType);
  PyModule_AddObject(m, "Array", (PyObject*) &mxArrayType);

  PyModule_AddIntMacro(m, mxREAL);
  PyModule_AddIntMacro(m, mxCOMPLEX);
  PyModule_AddIntMacro(m, mxUNKNOWN_CLASS);
  PyModule_AddIntMacro(m, mxCELL_CLASS);
  PyModule_AddIntMacro(m, mxSTRUCT_CLASS);
  PyModule_AddIntMacro(m, mxLOGICAL_CLASS);
  PyModule_AddIntMacro(m, mxCHAR_CLASS);
  PyModule_AddIntMacro(m, mxDOUBLE_CLASS);
  PyModule_AddIntMacro(m, mxSINGLE_CLASS);
  PyModule_AddIntMacro(m, mxINT8_CLASS);
  PyModule_AddIntMacro(m, mxUINT8_CLASS);
  PyModule_AddIntMacro(m, mxINT16_CLASS);
  PyModule_AddIntMacro(m, mxUINT16_CLASS);
  PyModule_AddIntMacro(m, mxINT32_CLASS);
  PyModule_AddIntMacro(m, mxUINT32_CLASS);
  PyModule_AddIntMacro(m, mxINT64_CLASS);
  PyModule_AddIntMacro(m, mxUINT64_CLASS);
  PyModule_AddIntMacro(m, mxFUNCTION_CLASS);

  mxmodule = m;

  #if PY_VERSION_HEX >= PY3K_VERSION_HEX
  return m;
  #endif
}
