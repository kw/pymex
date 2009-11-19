#include "pymex.h"
#include <mex.h>

/*
  FIXME: Most of these shared functions were developed before the Python-side
  of the bridge had its foundations. They throw MATLAB errors with appropriate
  messages. This is probably suboptimal if the caller is from the Python side
  of the bridge. Need to make the errors generic, or possibly just use Python
  exceptions. That sounds good, actually.
*/


/*
  box_by_type - Given a python object, will arrange for a MATLAB object
  of an appropriate type to be instantiated to hold the pointer. To allow
  the user to customize the object's MATLAB-side behavior, a hierarchy of
  MATLAB packages and classes exists in the +py/+types directory. The 
  PyObject's type is asked for its method resolution order to determine
  what order to check for MATLAB-equivalent classes. 

  The hierarchy should look something like py.types.modulename.classname.
  I haven't played around with more deeply nested structures, so that's
  a potential source of problems - we may need to check the __package__
  attribute as well. 

  Naturally, names must be normalized to obey MATLAB naming conventions.
  Presently the only thing done to enforce this is to strip underscores from
  both ends of the module name. 

  In the event that nothing in the type's mro list matches the available
  wrapper classes, py.types.builtin.object is used even if this type does
  not descend from object. If we were passed a NULL, we happily wrap it
  with py.types.voidptr, the base class of our wrapper hierarchy.

  Note that the only requirement on the wrapper is that running 'which'
  on 'py.types.modulename.classname' returns something, and that the indicated
  thing can be called with no arguments. So it may be a class, or it may be a
  simple m-function that returns a class instance. 

  box_by_type doesn't actually insert the PyObject's pointer into the result,
  it just instantiates the MATLAB wrapper. 
*/
#define MAX_MXTYPE_NAME_SIZE 512
mxArray* box_by_type(PyObject* pyobj) {
  static char mlname[MAX_MXTYPE_NAME_SIZE] = {0};
  static char* package = "py.types.%s.%s";
  PYMEX_DEBUG("Trying to box %p\n", pyobj);
  mxArray* box = NULL;
  mxArray* mxname;
  mxArray* which;
  int ret = -1;
  if (!pyobj) {
    ret = mexCallMATLAB(1,&box,0,NULL,PYMEX_MATLAB_VOIDPTR);
  }
  else {
    PyObject* type = (PyObject*) pyobj->ob_type;
    PyObject* mro;
    if (PyType_Check(pyobj)) {
      /* Calling type on a type gives us back type, and
	 calling mro on type doesn't work quite the same,
	 so we special case this one. We don't actually have
	 a wrapper type to wrap type, but if we did, it would.
	 Yo dawg.
      */
      PYMEX_DEBUG("Object is a type...\n");
      mro = Py_BuildValue("(O)", &PyType_Type);
    }
    else {
      PYMEX_DEBUG("Object is not a type...\n");
      mro = PyObject_CallMethod(type, "mro", "()");
    }
    Py_ssize_t len = PySequence_Length(mro);
    Py_ssize_t i;
    for (i=0; i<len && ret; i++) {
      PyObject* item = PySequence_GetItem(mro, i);
      if (item == pyobj) {
	PYMEX_DEBUG("Pointers match!?\n"); 
      }
      else if (!item) {
	PYMEX_DEBUG("Item is null!?\n");
      }
      else {
	PYMEX_DEBUG("Ok, getting name...\n");
      }
      PyObject* modname = PyObject_GetAttrString(item, "__module__");
      PyObject* cleanmodname = PyObject_CallMethod(modname, "strip", "s", "_");
      PyObject* name = PyObject_GetAttrString(item, "__name__");
      snprintf(mlname, MAX_MXTYPE_NAME_SIZE, package, PyBytes_AsString(cleanmodname), PyBytes_AsString(name));
      PYMEX_DEBUG("Checking for %s...\n", mlname);
      Py_DECREF(name);
      Py_DECREF(cleanmodname);
      Py_DECREF(modname);
      Py_DECREF(item);
      mxname = mxCreateString(mlname);
      mexCallMATLAB(1,&which,1,&mxname,"which");
      mxDestroyArray(mxname);
      if (mxGetNumberOfElements(which) > 0) {
	PYMEX_DEBUG("%s looks good, trying it...\n", mlname);
	ret = mexCallMATLAB(1,&box,0,NULL,mlname);
      }
      mxDestroyArray(which);
    }
    Py_DECREF(mro);
    if (ret) { /* none found, use sane default */
      PYMEX_DEBUG("No reasonable box found.\n");
      ret = mexCallMATLAB(1,&box,0,NULL,PYMEX_MATLAB_PYOBJECT);
    }
  }
  if (ret || !box) {
    PYMEX_DEBUG("Unable to find " PYMEX_MATLAB_PYOBJECT);
    PyErr_Format(PyExc_RuntimeError,"Unable to find %s", PYMEX_MATLAB_PYOBJECT);
    return NULL;
  }
  PYMEX_DEBUG("Returning box\n");
  return box;
}

/* boxes the object, stealing the reference */
mxArray* box (PyObject* pyobj) {
  mxArray* boxed = NULL;
  if (!pyobj) {
    PYMEX_DEBUG("Boxing null object.");
  }
  boxed = box_by_type(pyobj);
  if (!boxed) return NULL;
  if (pyobj) mexLock();
  mxArray* ptr_field = mxGetProperty(boxed, 0, "pointer");
  void** ptr = mxGetData(ptr_field);
  *ptr = (void*) pyobj;
  mxSetProperty(boxed, 0, "pointer", ptr_field);
  return boxed;
}

/* Box a borrowed reference (increfs it first) */
mxArray* boxb (PyObject* pyobj) {
  Py_XINCREF(pyobj);
  return box(pyobj);
}

/* Unboxes an object, returning a borrowed reference */
PyObject* unbox (const mxArray* mxobj) {  
  if (!mxobj) return PyErr_Format(PyExc_RuntimeError, "Can't unbox from null pointer");
  if (mxIsPyNull(mxobj)) {
    PYMEX_DEBUG("Unboxed a null object.");
    return PyErr_Format(PyExc_RuntimeError, "Unboxed pointer is null.");
  }
  else {
    void** ptr = mxGetData(mxGetProperty(mxobj, 0, "pointer"));
    return (PyObject*) *ptr;
  }
}

/* Unboxes an object, returning a new reference */
PyObject* unboxn (const mxArray* mxobj) {
  PyObject* pyobj;
  if (mxIsPyObject(mxobj)) {
    pyobj = unbox(mxobj);
    Py_XINCREF(pyobj);
  }
  else {
    pyobj = Any_mxArray_to_PyObject(mxobj);
  }
  return pyobj;
}

/* Returns true if the wrapper's pointer is NULL. 
   Only pass it voidptr (and subclasses thereof), as
   it does not check this before doing the mxGetProperty
 */
bool mxIsPyNull (const mxArray* mxobj) {
  void** ptr = mxGetData(mxGetProperty(mxobj, 0, "pointer"));
  return !*ptr;
}

/* Asks the MATLAB interpreter if the object isa subclass of voidptr.
   This includes null pointers and things not descended from object.
 */
bool mxIsPyObject(const mxArray* mxobj) {
  mxArray* boolobj;
  mxArray* args[2];
  args[0] = (mxArray*) mxobj;
  args[1] = mxCreateString(PYMEX_MATLAB_VOIDPTR);
  mexCallMATLAB(1,&boolobj,2,args,"isa");
  mxDestroyArray(args[1]);
  return mxIsLogicalScalarTrue(boolobj);
}

PyObject* mxChar_to_PyBytes(const mxArray* mxchar) {
  if (!mxchar || !mxIsChar(mxchar))
    mexErrMsgTxt("Input isn't a mxChar");
  char* tempstring = mxArrayToString(mxchar);
  if (!tempstring)
    mexErrMsgTxt("Couldn't stringify mxArray for some reason.");
  PyObject* pystr = PyBytes_FromString(tempstring);
  mxFree(tempstring);
  if (!pystr)
    mexErrMsgTxt("Couldn't convert from string to PyBytes");
  return pystr;
}

mxArray* PyBytes_to_mxChar(PyObject* pystr) {
  if (!pystr || !PyBytes_Check(pystr))
    mexErrMsgTxt("Input isn't a PyBytes");
  char* tempstring = PyBytes_AsString( pystr);
  mxArray* mxchar = mxCreateString(tempstring);
  return mxchar;
}

mxArray* PyObject_to_mxChar(PyObject* pyobj) {
  if (pyobj) {
    PyObject* pystr = PyObject_Str(pyobj);
    mxArray* mxchar = PyBytes_to_mxChar(pystr);
    Py_DECREF(pystr);
    return mxchar;
  } 
  else {
    PyErr_Format(PyExc_RuntimeError, "Can't convert NULL to string");
    return NULL;
  }
}

mxArray* PyObject_to_mxLogical(PyObject* pyobj) {
  return mxCreateLogicalScalar(PyObject_IsTrue(pyobj));
}

/*
PyObject* mxNonScalar_to_PyList(const mxArray* mxobj) {
  mwSize numel = mxGetNumberOfElements(mxobj);
  PyObject* pyobj = PyList_New(numel);
  mwSize i;
  mxClassID type = mxGetClassID(mxobj);
  PyObject* val;
  for (i=0; i<numel; i++) {
    switch (type) {
    case mxCELL_CLASS:
      val = Any_mxArray_to_PyObject(mxGetCell(mxobj, i)); break;
    default:
      val = mxNumber_to_PyObject(mxobj, i); break;
    }
    PyList_SetItem(pyobj, i, val);
  }
  return pyobj;
}
*/

PyObject* mxCell_to_PyTuple(const mxArray* mxobj) {
  mwSize numel = mxGetNumberOfElements(mxobj);
  PyObject* pyobj = PyTuple_New(numel);
  mwSize i;
  for (i=0; i<numel; i++) {
    PyTuple_SetItem(pyobj, i, Any_mxArray_to_PyObject(mxGetCell(mxobj, i)));
  }
  return pyobj;
}

PyObject* mxCell_to_PyTuple_recursive(const mxArray* mxobj) {
  mwSize numel = mxGetNumberOfElements(mxobj);
  PyObject* pyobj = PyTuple_New(numel);
  mwSize i;
  for (i=0; i<numel; i++) {
    mxArray* item = mxGetCell(mxobj, i);
    PyObject* pyitem;
    if (mxIsCell(item))
      pyitem = mxCell_to_PyTuple_recursive(item);
    else
      pyitem = Any_mxArray_to_PyObject(item);
    PyTuple_SetItem(pyobj, i, pyitem);
  }
  return pyobj;
}

/*
PyObject* mxNumber_to_PyObject(const mxArray* mxobj, mwIndex index) {
  void* ptr = mxGetData(mxobj);
#define DECODE(fmt, type) return Py_BuildValue(#fmt, *(((type *) ptr)+index))
  switch (mxGetClassID(mxobj)) {
  case mxLOGICAL_CLASS:
    if (*(((mxLogical*) ptr)+index))
      Py_RETURN_TRUE;
    else
      Py_RETURN_FALSE;
  case mxINT8_CLASS:     
    DECODE(b,signed char);
  case mxUINT8_CLASS:
    DECODE(B,unsigned char);
  case mxINT16_CLASS:
    DECODE(h,short);
  case mxUINT16_CLASS:
    DECODE(H,unsigned short);
  case mxINT32_CLASS:
    DECODE(i,int);
  case mxUINT32_CLASS:
    DECODE(I,unsigned int);
  case mxINT64_CLASS:
    DECODE(L, long long);
  case mxUINT64_CLASS:
    DECODE(K, unsigned long long);
  case mxDOUBLE_CLASS:
    DECODE(d, double);
  case mxSINGLE_CLASS:
    DECODE(f, float);
  default:
    mexErrMsgIdAndTxt("pymex:convertMXtoPY", "Don't know how to convert %s", 
		      mxGetClassName(mxobj));
    return NULL;
  }
#undef DECODE
}
*/

mxArray* PySequence_to_mxCell(PyObject* pyobj) {
  PyObject* item;
  Py_ssize_t len = PySequence_Length(pyobj);
  mxArray* mxcell = mxCreateCellMatrix(1,len);
  mexMakeArrayPersistent(mxcell);
  mwIndex i;
  for (i=0; i<len; i++) {
    item = PySequence_GetItem(pyobj, i);
    mxSetCell(mxcell, i, Any_PyObject_to_mxArray(item));
    Py_DECREF(item);
  }
  return mxcell;
}
/*
int PySequence_is_Numeric(PyObject* pyobj) {
  PyObject* item;
  Py_ssize_t len = PySequence_Length(pyobj);
  bool hasfloat = 0;
  bool hasint = 0;
  bool hasother = 0;
  Py_ssize_t i;
  for (i=0; i<len; i++) {
    item = PySequence_GetItem(pyobj, i);
    if (!PyNumber_Check(item)) {
      Py_DECREF(item);
      return 0;
    }
    if (PyLong_Check(item) || PyLong_Check(item))
      hasint = 1;
    else if (PyFloat_Check(item))
      hasfloat = 1;
    else
      hasother = 1;
  }
  if (hasother)
    mexWarnMsgTxt("Sequence has numeric items that don't check as ints, "
		  "longs, or floats, and might not handle right.");
  if (hasfloat)
    return 2;
  else if (hasint)
    return 1;
  else 
    return -1;
}
*/

PyObject* Any_mxArray_to_PyObject(const mxArray* mxobj) {
  if (mxIsPyObject(mxobj)) {
    PyObject* pyobj = unbox(mxobj);
    Py_XINCREF(pyobj);
    return pyobj;
  }
  else if (mxIsChar(mxobj)) {
    return mxChar_to_PyBytes(mxobj);
  }
  else {
    return Py_mxArray_New((mxArray*) mxobj,1);
  }
}

mxArray* Any_PyObject_to_mxArray(PyObject* pyobj) {
  if (!pyobj)
    return box(pyobj); /* Null pointer */
  else if (PyBytes_Check(pyobj))
    return PyBytes_to_mxChar(pyobj);
  else if (Py_mxArray_Check(pyobj))
    return mxArrayPtr(pyobj);
  else
    return boxb(pyobj);
}

/* TODO: Low priority, but this depends on libmex. If we try to make an
   external module later, this will need to function to some extent.
   It could operate at low capability by using a hardcoded list for
   the builtins and assuming that anything else has _object as a super.
 */
PyObject* Calculate_matlab_mro(mxArray* mxobj) {
  mxArray* argin[3];
  argin[0] = mxobj;
  argin[1] = mxCreateLogicalScalar(1); /* addvirtual=true */
  argin[2] = mxCreateLogicalScalar(1); /* autosplit=true */
  mxArray* argout[1] = {NULL};
  mexSetTrapFlag(1);
  int err = mexCallMATLAB(1, argout, 3, argin, "mro");
  if (err) return PyErr_Format(PyExc_RuntimeError, "MATLAB error while trying to run 'mro' function.");
  else {
    PyObject* retval = mxCell_to_PyTuple_recursive(argout[0]);
    mxDestroyArray(argout[0]);
    mxDestroyArray(argin[1]);
    mxDestroyArray(argin[2]);
    return retval;
  }
}

/* Attempts to locate an appropriate subclass of mx.Array using the mltypes package.
   If for some reason this fails, mx.Array is returned instead.
 */
PyObject* Find_mltype_for(mxArray* mxobj) {
  #define GET_MX_ARRAY_CLASS PyObject_GetAttrString(mxmodule, "Array")
  PyObject *newclass, *mrolist, *mltypes, *findtype;
  newclass = mrolist = mltypes = findtype = NULL;
  mrolist = Calculate_matlab_mro(mxobj);
  if (!mrolist) {
    mexPrintf("failed to get mro list\n");
    PyErr_Clear();
    newclass = GET_MX_ARRAY_CLASS;
    goto findtypes_error;
  }
  mltypes = PyImport_ImportModule("mltypes");
  if (!mltypes) {
    mexPrintf("failed to import mltypes\n");
    PyErr_Clear();
    newclass = GET_MX_ARRAY_CLASS;
    goto findtypes_error;
  }
  findtype = PyObject_GetAttrString(mltypes, "_findtype");
  if (!findtype) {
    mexPrintf("failed to find _findtype\n");
    PyErr_Clear();
    newclass = GET_MX_ARRAY_CLASS;
    goto findtypes_error;
  }
  /* I'm not sure why we need *another* tuple around this, but apparently
     CallFunction kills the outermost tuple here */
  mrolist = PyTuple_Pack(1, mrolist);
  newclass = PyObject_CallFunction(findtype, "O", mrolist);
  if (!newclass) {
    mexPrintf("failed to call _findtype\n");
    PyErr_Clear();
    newclass = GET_MX_ARRAY_CLASS;
    goto findtypes_error;
  }
  findtypes_error:
  Py_XDECREF(findtype);
  Py_XDECREF(mrolist);
  Py_XDECREF(mltypes);
  return newclass;
}


PyObject* Py_mxArray_New(mxArray* mxobj, bool duplicate) {
  mxArray* copy;
  if (duplicate) {
    copy = mxDuplicateArray(mxobj);
    mexMakeArrayPersistent(copy);
  }
  else {
    copy = mxobj;
  }
  PyObject* mxptr = mxArrayPtr_New(copy);
  PyObject* args = PyTuple_New(0);
  PyObject* kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "mxpointer", mxptr);
  PyObject* arraycls = Find_mltype_for(copy);
  /* TODO: There is probably a better way to do this... */
  PyObject* ret = PyObject_Call(arraycls, args, kwargs);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_DECREF(mxptr);
  Py_DECREF(arraycls);
  return ret;
}

int Py_mxArray_Check(PyObject* pyobj) {
  return PyObject_IsInstance(pyobj, PyObject_GetAttrString(mxmodule, "Array"));
}

#if PYMEX_USE_NUMPY

mxArray* PyObject_to_mxDouble(PyObject* pyobj) {
  PyArray_Descr* type = PyArray_DescrFromType(NPY_FLOAT64);
  PyObject* array = PyArray_FromAny(pyobj, type, 0, 0, NPY_FARRAY, NULL);
  if (!array) return NULL;
  mxArray* out = PyArray_to_mxArray(array);
  Py_DECREF(array);
  return out;
}

mxArray* PyObject_to_mxLong(PyObject* pyobj) {
  PyArray_Descr* type = PyArray_DescrFromType(NPY_INT64);
  PyObject* array = PyArray_FromAny(pyobj, type, 0, 0, NPY_FARRAY, NULL);
  if (!array) return NULL;
  mxArray* out = PyArray_to_mxArray(array);
  Py_DECREF(array);
  return out;
}
#else

mxArray* PyObject_to_mxDouble(PyObject* pyobj) {
  if (PySequence_Check(pyobj)) {
    Py_ssize_t len = PySequence_Length(pyobj);
    mxArray* mxval = mxCreateDoubleMatrix(1, len, mxREAL);
    mexMakeArrayPersistent(mxval);
    mwIndex i;
    double* ptr = (double*) mxGetData(mxval);
    PyObject* item;
    for (i=0; i<len; i++) {
      item = PySequence_ITEM(pyobj, i);
      ptr[i] = PyFloat_AsDouble(item);
      if (PyErr_Occurred()) return NULL;
      Py_DECREF(item);
    }
    return mxval;
  }
  else {
    double val = PyFloat_AsDouble(pyobj);
    if (PyErr_Occurred()) return NULL;
    mxArray* mxval = mxCreateDoubleScalar(val);
    mexMakeArrayPersistent(mxval);
    return mxval;
  }
}

mxArray* PyObject_to_mxLong(PyObject* pyobj) {
  if (PySequence_Check(pyobj)) {
    Py_ssize_t len = PySequence_Length(pyobj);
    mxArray* mxval = mxCreateNumericMatrix(1, len, mxINT64_CLASS, mxREAL);
    mexMakeArrayPersistent(mxval);
    mwIndex i;
    long* ptr = (long*) mxGetData(mxval);
    PyObject* item;
    for (i=0; i<len; i++) {
      item = PySequence_ITEM(pyobj, i);
      ptr[i] = PyLong_AsLong(item);
      if (PyErr_Occurred()) return NULL;
      Py_DECREF(item);
    }
    return mxval;
  }
  else {
    double val = PyLong_AsLong(pyobj);
    if (PyErr_Occurred()) return NULL;
    /* there is no mxCreateNumericScalar */
    mxArray* mxval = mxCreateNumericMatrix(1, 1, mxINT64_CLASS, mxREAL);
    mexMakeArrayPersistent(mxval);
    long* ptr = (long*) mxGetData(mxval);
    ptr[0] = val;
    return mxval;
  }
}

#endif


#if PYMEX_USE_NUMPY
/* There is likely some problem with these conversions. Try not to use this. */
int mxClassID_to_PyArrayType(mxClassID mxclass) {
  switch (mxclass) {
  case mxLOGICAL_CLASS: return NPY_BOOL;
  case mxCHAR_CLASS: return NPY_STRING;
  case mxINT8_CLASS: return NPY_INT8;
  case mxUINT8_CLASS: return NPY_UINT8;
  case mxINT16_CLASS: return NPY_INT16;
  case mxUINT16_CLASS: return NPY_UINT16;
  case mxINT32_CLASS: return NPY_INT32;
  case mxUINT32_CLASS: return NPY_UINT32;
  case mxINT64_CLASS: return NPY_INT64;
  case mxUINT64_CLASS: return NPY_UINT64;
  case mxSINGLE_CLASS: return NPY_FLOAT32;
  case mxDOUBLE_CLASS: return NPY_FLOAT64;
  default: return NPY_OBJECT;
  }
}

mxClassID PyArrayType_to_mxClassID(int pytype) {
  switch (pytype) {
  case NPY_BOOL: return mxLOGICAL_CLASS;
  case NPY_STRING: return mxCHAR_CLASS;
  case NPY_INT8: return mxINT8_CLASS;
  case NPY_UINT8: return mxUINT8_CLASS;
  case NPY_INT16: return mxINT16_CLASS;
  case NPY_UINT16: return mxUINT16_CLASS;
  case NPY_INT32: return mxINT32_CLASS;
  case NPY_UINT32: return mxUINT32_CLASS;
  case NPY_INT64: return mxINT64_CLASS;
  case NPY_UINT64: return mxUINT64_CLASS;
  case NPY_FLOAT32: return mxSINGLE_CLASS;
  case NPY_FLOAT64: return mxDOUBLE_CLASS;
  default: 
    mexErrMsgIdAndTxt("pymex:badtype","Could not determine appropriate type for %d", pytype);
    return mxUNKNOWN_CLASS;
  }
}

PyObject* mxArray_to_PyArray(const mxArray* mxobj, bool duplicate) {
  PyObject* base = NULL;
  mxArray* ptr = (mxArray*) mxobj;
  if (duplicate) {
    base = Py_mxArray_New(mxobj,1);
    ptr = mxArrayPtr(base);
  }
  void* data = mxGetData(ptr);
  int realnd = (int) mxGetNumberOfDimensions(ptr);
  const mwSize* mxdims = mxGetDimensions(ptr);
  int nd = realnd;
#if PYMEX_SQUEEZE_SMALL_ARRAYS
  if (realnd > 2)
    nd = realnd;
  else if (mxdims[0] == 1 && mxdims[1] == 1)
    nd = 0;
  else if (mxdims[0] == 1 || mxdims[1] == 1)
    nd = 1;  
  else
    nd = 2;
  npy_intp *dims = NULL;
  if (nd >= 2) {
    dims = malloc(nd*sizeof(npy_intp));
    int i;
    for (i=0; i<nd; i++) {
      dims[i] = (npy_intp) mxdims[i];
    }
  }  
  else if (nd == 1) {
    dims = malloc(sizeof(npy_intp));
    if (mxdims[0] == 1)
      dims[0] = mxdims[1];
    else
      dims[0] = mxdims[0];
  }
#else
  npy_intp dims[nd];
  int i;
  for (i=0; i<nd; i++) {
      dims[i] = (npy_intp) mxdims[i];
  }
#endif /* PYMEX_SQUEEZE_SMALL_ARRAYS */
  int typenum = mxClassID_to_PyArrayType(mxGetClassID(ptr));
  int itemsize = (int) mxGetElementSize(ptr);
  int flags = NPY_FARRAY;
  PyArrayObject* array = (PyArrayObject*)
    PyArray_New(&PyArray_Type, nd, dims, typenum, NULL, data,
		itemsize, flags, NULL);
  array->base = base;
#if PYMEX_SQUEEZE_SMALL_ARRAYS
  free(dims);
#endif
  return (PyObject*) array;
}

mxArray* PyArray_to_mxArray(PyObject* pyobj) {
  pyobj = PyArray_EnsureArray(pyobj);
  mxClassID class = PyArrayType_to_mxClassID(PyArray_TYPE(pyobj));
  int realnd = PyArray_NDIM(pyobj);
  int nd;
  if (realnd < 2) 
    nd = 2;
  else
    nd = realnd;		
  npy_intp* ndims = PyArray_DIMS(pyobj);
  mwSize mxdims[nd];
  mwSize i = 0;
  mxdims[0] = 1;
  switch (realnd) {
  case 0: mxdims[1] = 1; break;
  case 1: mxdims[1] = (mwSize) ndims[0]; break;
  default:
    for (i=0; i<nd; i++) {
      mxdims[i] = (mwSize) ndims[i];
    }
  }      
  mxArray* result = mxCreateNumericArray(nd, mxdims, class, mxREAL);
  mexMakeArrayPersistent(result);
  PyObject* pyresult = mxArray_to_PyArray(result, false);
  PyArray_CopyInto((PyArrayObject*) pyresult, (PyArrayObject*) pyobj);
  Py_DECREF(pyresult);
  return result;
}
#endif /* PYMEX_USE_NUMPY */

char mxClassID_to_Numpy_Typekind(mxClassID mxclass) {
  switch (mxclass) {
  case mxLOGICAL_CLASS: return 'b';
  case mxCHAR_CLASS: return 'S';
  case mxINT8_CLASS: 
  case mxINT16_CLASS: 
  case mxINT32_CLASS: 
  case mxINT64_CLASS: 
  case mxUINT8_CLASS: return 'i'; 
  case mxUINT16_CLASS: 
  case mxUINT32_CLASS: 
  case mxUINT64_CLASS: return 'u'; 
  case mxSINGLE_CLASS: 
  case mxDOUBLE_CLASS: return 'f';
  default: return 'V';
  }
}

mxArray* mxArrayPtr(PyObject* pyobj) {
  PyObject* ptr;
  if (PyCObject_Check(pyobj)) {
    ptr = pyobj;
  }
  else {
    mxArrayObject* mxobj = (mxArrayObject*) pyobj;
    ptr = mxobj->mxptr;
  }
  if (!ptr || PyCObject_GetDesc(ptr) != mxmodule) {
    PyErr_Format(PyExc_RuntimeError, "mxptr desc does not match mxmodule");
    return NULL;
  }
  return (mxArray*) PyCObject_AsVoidPtr(ptr);
}

static void _mxArrayPtr_destructor(void* mxobj, void* desc) {
  Py_XDECREF((PyObject*) desc);  
  if (mxobj) mxDestroyArray((mxArray*) mxobj);
}

PyObject* mxArrayPtr_New(mxArray* mxobj) {
  if (!mxmodule)
    return PyErr_Format(PyExc_RuntimeError, "mxmodule not yet initialized");
  PERSIST_ARRAY(mxobj);
  Py_INCREF(mxmodule);
  return PyCObject_FromVoidPtrAndDesc(mxobj, mxmodule, _mxArrayPtr_destructor);
}

int mxArrayPtr_Check(PyObject* obj) {
  return (obj && PyCObject_Check(obj) && PyCObject_GetDesc(obj) == mxmodule);
}
