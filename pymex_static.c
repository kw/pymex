#define PRIVATE static

/* PYMEX_SQUEEZE_SMALL_ARRAYS
   MATLAB does not allow arrays to have fewer than 2 dimensions.
   Numpy, on the other hand, allows even 1-d or even 0-d arrays (scalars).
   When converting from Numpy to MATLAB arrays, we must naturally increase
   the number of dimensions for small-dimensioned arrays, but going the other
   direction isn't necessarily clear.
   If PYMEX_SQUEEZE_SMALL_ARRAYS is true, pymex will detect singleton dimensions
   of a 2D array and remove them. This might make broadcasting easier, but
   it also causes vectors to forget their orientation. 
 */
#ifndef PYMEX_SQUEEZE_SMALL_ARRAYS
#define PYMEX_SQUEEZE_SMALL_ARRAYS 1
#endif

PRIVATE char* PyMX_Marker = "PyMXObject";
PRIVATE mxArray* box(PyObject* pyobj);
PRIVATE mxArray* boxb(PyObject* pyobj);
PRIVATE PyObject* unbox (const mxArray* mxobj);
PRIVATE PyObject* unboxn (const mxArray* mxobj);
PRIVATE bool mxIsPyNull (const mxArray* mxobj);
PRIVATE bool mxIsPyObject(const mxArray* mxobj);
PRIVATE mxArray* PyObject_to_mxLogical(PyObject* pyobj);
PRIVATE PyObject* mxChar_to_PyString(const mxArray* mxchar);
PRIVATE mxArray* PyString_to_mxChar(PyObject* pystr);
PRIVATE mxArray* PyObject_to_mxChar(PyObject* pyobj);
PRIVATE mxArray* PySequence_to_mxCell(PyObject* pyobj);
PRIVATE mxArray* PyObject_to_mxDouble(PyObject* pyobj);
PRIVATE mxArray* PyObject_to_mxLong(PyObject* pyobj);
PRIVATE PyObject* Any_mxArray_to_PyObject(const mxArray* mxobj);
PRIVATE mxArray* Any_PyObject_to_mxArray(PyObject* pyobj);
PRIVATE bool PyMXObj_Check(PyObject* pyobj);
PRIVATE PyObject* PyCObject_from_mxArray(const mxArray* mxobj);
PRIVATE PyObject* mxArray_to_PyArray(const mxArray* mxobj, bool duplicate);
PRIVATE mxArray* PyArray_to_mxArray(PyObject* pyobj);

PRIVATE mxArray* box_by_type(PyObject* pyobj) {
  PYMEX_DEBUG("Trying to box %p\n", pyobj);
  char* package = "pytypes.%s";
  char mlname[128] = {0};
  mxArray* box = NULL;
  mxArray* mxname;
  mxArray* which;
  int ret = -1;
  if (!pyobj) {
    ret = mexCallMATLAB(1,&box,0,NULL,"pytypes.voidptr");
  }
  else {
    PyObject* type = (PyObject*) pyobj->ob_type;
    PyObject* mro;
    if (PyType_Check(pyobj)) {
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
      PyObject* name = PyObject_GetAttrString(item, "__name__");
      snprintf(mlname, 128, package, PyString_AsString(name));
      PYMEX_DEBUG("Checking for %s...\n", mlname);
      Py_DECREF(name);
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
      ret = mexCallMATLAB(1,&box,0,NULL,"pytypes.object");
    }
  }
  if (ret || !box)
    mexErrMsgIdAndTxt("pymex:NoBoxes","Unable to find pytypes.object");
  PYMEX_DEBUG("Returning box\n");
  return box;
}

               
PRIVATE mxArray* box (PyObject* pyobj) {
  mxArray* boxed = NULL;
  if (!pyobj) {
    PYMEX_DEBUG("Attempted to box null object.");
  }
  boxed = box_by_type(pyobj);
  if (pyobj) {
    mexLock();
    mxArray* ptr_field = mxGetProperty(boxed, 0, "pointer");
    void** ptr = mxGetData(ptr_field);
    *ptr = (void*) pyobj;
    mxSetProperty(boxed, 0, "pointer", ptr_field);
  }
  return boxed;
}

PRIVATE mxArray* boxb (PyObject* pyobj) {
  Py_XINCREF(pyobj);
  return box(pyobj);
}

PRIVATE PyObject* unbox (const mxArray* mxobj) {  
  if (mxIsPyNull(mxobj)) {
    PYMEX_DEBUG("Attempt to unbox null object.");
    return NULL;
  }
  else {
    void** ptr = mxGetData(mxGetProperty(mxobj, 0, "pointer"));
    return (PyObject*) *ptr;
  }
}

PRIVATE PyObject* unboxn (const mxArray* mxobj) {
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

PRIVATE bool mxIsPyNull (const mxArray* mxobj) {
  void** ptr = mxGetData(mxGetProperty(mxobj, 0, "pointer"));
  return !*ptr;
}

PRIVATE bool mxIsPyObject(const mxArray* mxobj) {
  mxArray* boolobj;
  mxArray* args[2];
  args[0] = (mxArray*) mxobj;
  args[1] = mxCreateString("pytypes.object");
  mexCallMATLAB(1,&boolobj,2,args,"isa");
  mxDestroyArray(args[1]);
  return mxIsLogicalScalarTrue(boolobj);
}

PRIVATE PyObject* mxChar_to_PyString(const mxArray* mxchar) {
  if (!mxchar || !mxIsChar(mxchar))
    mexErrMsgTxt("Input isn't a mxChar");
  char* tempstring = mxArrayToString(mxchar);
  if (!tempstring)
    mexErrMsgTxt("Couldn't stringify mxArray for some reason.");
  PyObject* pystr = PyString_FromString(tempstring);
  mxFree(tempstring);
  if (!pystr)
    mexErrMsgTxt("Couldn't convert from string to PyString");
  return pystr;
}

PRIVATE mxArray* PyString_to_mxChar(PyObject* pystr) {
  if (!pystr || !PyString_Check(pystr))
    mexErrMsgTxt("Input isn't a PyString");
  char* tempstring = PyString_AsString( pystr);
  mxArray* mxchar = mxCreateString(tempstring);
  return mxchar;
}

PRIVATE mxArray* PyObject_to_mxChar(PyObject* pyobj) {
  mxArray* mxchar;
  if (pyobj) {
    PyObject* pystr = PyObject_Str(pyobj);
    mxchar = PyString_to_mxChar(pystr);
    Py_DECREF(pystr);
  } 
  else {
    mxchar = mxCreateString("<NULL>");
  }
  return mxchar;
}

PRIVATE mxArray* PyObject_to_mxLogical(PyObject* pyobj) {
  return mxCreateLogicalScalar(PyObject_IsTrue(pyobj));
}

/*
PRIVATE PyObject* mxNonScalar_to_PyList(const mxArray* mxobj) {
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

PRIVATE PyObject* mxCell_to_PyTuple(const mxArray* mxobj) {
  mwSize numel = mxGetNumberOfElements(mxobj);
  PyObject* pyobj = PyTuple_New(numel);
  mwSize i;
  for (i=0; i<numel; i++) {
    PyTuple_SetItem(pyobj, i, Any_mxArray_to_PyObject(mxGetCell(mxobj, i)));
  }
  return pyobj;
}
/*
PRIVATE PyObject* mxNumber_to_PyObject(const mxArray* mxobj, mwIndex index) {
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

PRIVATE mxArray* PyObject_to_mxDouble(PyObject* pyobj) {
  PyArray_Descr* type = PyArray_DescrFromType(NPY_FLOAT64);
  PyObject* array = PyArray_FromAny(pyobj, type, 0, 0, NPY_FARRAY, NULL);
  if (!array) return NULL;
  mxArray* out = PyArray_to_mxArray(array);
  Py_DECREF(array);
  return out;
}

PRIVATE mxArray* PyObject_to_mxLong(PyObject* pyobj) {
  PyArray_Descr* type = PyArray_DescrFromType(NPY_INT64);
  PyObject* array = PyArray_FromAny(pyobj, type, 0, 0, NPY_FARRAY, NULL);
  if (!array) return NULL;
  mxArray* out = PyArray_to_mxArray(array);
  Py_DECREF(array);
  return out;
}

PRIVATE mxArray* PySequence_to_mxCell(PyObject* pyobj) {
  PyObject* item;
  Py_ssize_t len = PySequence_Length(pyobj);
  mxArray* mxcell = mxCreateCellMatrix(1,len);
  mwIndex i;
  for (i=0; i<len; i++) {
    item = PySequence_GetItem(pyobj, i);
    mxSetCell(mxcell, i, Any_PyObject_to_mxArray(item));
    Py_DECREF(item);
  }
  return mxcell;
}
/*
PRIVATE int PySequence_is_Numeric(PyObject* pyobj) {
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
    if (PyInt_Check(item) || PyLong_Check(item))
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

PRIVATE PyObject* Any_mxArray_to_PyObject(const mxArray* mxobj) {
  if (mxIsPyObject(mxobj)) {
    PyObject* pyobj = unbox(mxobj);
    Py_XINCREF(pyobj);
    return pyobj;
  }
  else if (mxIsChar(mxobj)) {
    return mxChar_to_PyString(mxobj);
  }
  else if (mxIsNumeric(mxobj) || mxIsLogical(mxobj)) {
    return PyArray_Return((PyArrayObject*) mxArray_to_PyArray(mxobj, true));
  }
  else if (mxIsCell(mxobj)) {
    return mxCell_to_PyTuple(mxobj);
  }
  else {
    return PyCObject_from_mxArray(mxobj);
  }
}

PRIVATE mxArray* Any_PyObject_to_mxArray(PyObject* pyobj) {
  PyObject* array = NULL;
  if (!pyobj)
    return box(pyobj); /* Null pointer */
  else if (PyString_Check(pyobj))
    return PyString_to_mxChar(pyobj);
  else if (PyArray_Check(pyobj))
    return PyArray_to_mxArray(pyobj);
  else if (PyBool_Check(pyobj))
    return PyObject_to_mxLogical(pyobj);
  else if (PyInt_Check(pyobj) || PyLong_Check(pyobj))
    return PyObject_to_mxLong(pyobj);
  else if (PyFloat_Check(pyobj))
    return PyObject_to_mxDouble(pyobj);
  else if (PyMXObj_Check(pyobj))
    return (mxArray*) PyCObject_AsVoidPtr(pyobj);
  else if (PyArray_HasArrayInterface(pyobj, array))
    return box(array);
  else if (PySequence_Check(pyobj))
    return PySequence_to_mxCell(pyobj);  
  else
    return boxb(pyobj);
}

PRIVATE bool PyMXObj_Check(PyObject* pyobj) {
  return PyCObject_Check(pyobj) && (PyCObject_GetDesc(pyobj) == (void*) PyMX_Marker);
}

PRIVATE void PyMXDestructor(void* mxobj, void* desc) {
  mxDestroyArray((mxArray*) mxobj);
}

PRIVATE PyObject* PyCObject_from_mxArray(const mxArray* mxobj) {
  mxArray* copy = mxDuplicateArray(mxobj);
  mexMakeArrayPersistent(copy);
  return PyCObject_FromVoidPtrAndDesc((void*) copy, (void*) PyMX_Marker, PyMXDestructor);
}

PRIVATE int mxClassID_to_PyArrayType(mxClassID mxclass) {
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

PRIVATE mxClassID PyArrayType_to_mxClassID(int pytype) {
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

PRIVATE PyObject* mxArray_to_PyArray(const mxArray* mxobj, bool duplicate) {
  PyObject* base = NULL;
  mxArray* ptr = (mxArray*) mxobj;
  if (duplicate) {
    base = PyCObject_from_mxArray(mxobj);
    ptr = PyCObject_AsVoidPtr(base);
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
#endif
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

PRIVATE mxArray* PyArray_to_mxArray(PyObject* pyobj) {
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
  PyObject* pyresult = mxArray_to_PyArray(result, false);
  PyArray_CopyInto((PyArrayObject*) pyresult, (PyArrayObject*) pyobj);
  Py_DECREF(pyresult);
  return result;
}


#undef PRIVATE
