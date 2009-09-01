#define PRIVATE

PRIVATE mxArray* mro (const PyObject* pyobj);
PRIVATE mxArray* box(const PyObject* pyobj);
PRIVATE mxArray* boxb(const PyObject* pyobj);
PRIVATE PyObject* unbox (const mxArray* mxobj);
PRIVATE PyObject* unboxn (const mxArray* mxobj);
PRIVATE bool mxIsPyNull (const mxArray* mxobj);
PRIVATE bool mxIsPyObject(const mxArray* mxobj);
PRIVATE PyObject* mxChar_to_PyString(const mxArray* mxchar);
PRIVATE mxArray* PyString_to_mxChar(const PyObject* pystr);
PRIVATE mxArray* PyObject_to_mxChar(const PyObject* pyobj);
PRIVATE mxArray* PySequence_to_mxDouble(const PyObject* pyobj);
PRIVATE mxArray* PySequence_to_mxLong(const PyObject* pyobj);
PRIVATE mxArray* PyObject_to_mxDouble(const PyObject* pyobj);
PRIVATE mxArray* PyObject_to_mxLong(const PyObject* pyobj);
PRIVATE PyObject* mxCell_to_PyObject(const mxArray* mxobj);
PRIVATE PyObject* mxNumber_to_PyObject(const mxArray* mxobj, mwIndex i);
PRIVATE PyObject* mxNonScalar_to_PyList(const mxArray* mxobj);
PRIVATE PyObject* Any_mxArray_to_PyObject(const mxArray* mxobj);
PRIVATE PyObject* PyDict_from_KW(const mxArray* kwargs);

PRIVATE mxArray* mro (const PyObject* pyobj) {
  mxArray* mrocell;
  if (!pyobj) {
    mrocell = mxCreateCellMatrix(1,1);
    mxSetCell(mrocell, 0, mxCreateString("null"));
  }
  else if (PyType_Check(pyobj)) {
    /* Lump all types together */
    mrocell = mxCreateCellMatrix(1,1);
    mxSetCell(mrocell, 0, mxCreateString("type"));
  }
  else {
    PyObject* type = PyObject_Type((PyObject*) pyobj);
    PyObject* mro = PyObject_CallMethod(type, "mro", "()");
    mwSize len = (mwSize) PySequence_Length(mro);
    mrocell = mxCreateCellMatrix(1,len);
    mwIndex i;
    for (i=0; i<len; i++) {
      PyObject* item = PySequence_GetItem(mro, i);
      PyObject* name = PyObject_GetAttrString(item, "__name__");
      mxSetCell(mrocell, i, mxCreateString(PyString_AsString(name)));
      Py_DECREF(name);
      Py_DECREF(item);
    }
  }
  return mrocell;
}

PRIVATE mxArray* box (const PyObject* pyobj) {
  mxArray* boxed = NULL;
  mxArray* args[1];
  if (!pyobj) {
    PYMEX_DEBUG("Attempted to box null object.");
  }
  args[0] = mro(pyobj);
  mexCallMATLAB(1,&boxed,1,args,"py.pointer_by_mro");
  if (pyobj) {
    mexLock();
    mxArray* ptr_field = mxGetProperty(boxed, 0, "pointer");
    void** ptr = mxGetData(ptr_field);
    *ptr = (void*) pyobj;
    mxSetProperty(boxed, 0, "pointer", ptr_field);
  }
  return boxed;
}

PRIVATE mxArray* boxb (const PyObject* pyobj) {
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
  args[1] = mxCreateString("py.types.object");
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

PRIVATE mxArray* PyString_to_mxChar(const PyObject* pystr) {
  if (!pystr || !PyString_Check(pystr))
    mexErrMsgTxt("Input isn't a PyString");
  char* tempstring = PyString_AsString((PyObject*) pystr);
  mxArray* mxchar = mxCreateString(tempstring);
  return mxchar;
}

PRIVATE mxArray* PyObject_to_mxChar(const PyObject* pyobj) {
  mxArray* mxchar;
  if (pyobj) {
    PyObject* pystr = PyObject_Str((PyObject*) pyobj);
    mxchar = PyString_to_mxChar(pystr);
    Py_DECREF(pystr);
  } 
  else {
    mxchar = mxCreateString("<NULL>");
  }
  return mxchar;
}

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

PRIVATE PyObject* mxCell_to_PyTuple(const mxArray* mxobj) {
  mwSize numel = mxGetNumberOfElements(mxobj);
  PyObject* pyobj = PyTuple_New(numel);
  mwSize i;
  for (i=0; i<numel; i++) {
    PyTuple_SetItem(pyobj, i, Any_mxArray_to_PyObject(mxGetCell(mxobj, i)));
  }
  return pyobj;
}

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
  }
#undef DECODE
}

PRIVATE PyObject* PyTuple_Single(const PyObject* pyobj) {
  PyObject* tuple = PyTuple_New(1);
  Py_INCREF(pyobj);
  PyTuple_SetItem(tuple,0,(PyObject*) pyobj);
  return tuple;
}

PRIVATE mxArray* PyObject_to_mxDouble(const PyObject* pyobj) {
  if (PySequence_Check((PyObject*) pyobj)) {
    return PySequence_to_mxDouble(pyobj);
  }
  else {
    PyObject* pyfloat = PyNumber_Float((PyObject*) pyobj);
    double scalar = PyFloat_AS_DOUBLE(pyfloat);
    Py_DECREF(pyfloat);
    return mxCreateDoubleScalar(scalar);
  }
}

PRIVATE mxArray* PyObject_to_mxLong(const PyObject* pyobj) {
  if (PySequence_Check((PyObject*) pyobj)) {
    return PySequence_to_mxLong(pyobj);
  }
  else {
    PyObject* pylong = PyNumber_Long((PyObject*) pyobj);
    long long scalar = PyLong_AsLongLong(pylong);
    Py_DECREF(pylong);
    mxArray* array = mxCreateNumericMatrix(1,1,mxINT64_CLASS, mxREAL);
    long long *data = mxGetData(array);
    *data = scalar;
    return array;
  }
}

PRIVATE mxArray* PySequence_to_mxDouble(const PyObject* pyobj) {
  PyObject* item;
  PyObject* pyfloat;
  Py_ssize_t len = PySequence_Length((PyObject*) pyobj);
  mxArray* rowvec = mxCreateDoubleMatrix(1,len, mxREAL);
  double* row = (double*) mxGetData(rowvec);
  mwIndex i;
  for (i=0; i<len; i++) {
    item = PySequence_GetItem((PyObject*) pyobj, i);
    pyfloat = PyNumber_Float(item);
    row[i] = PyFloat_AS_DOUBLE(pyfloat);
    Py_DECREF(pyfloat);
    Py_DECREF(item);
  }
  return rowvec;
}

PRIVATE mxArray* PySequence_to_mxLong(const PyObject* pyobj) {
  PyObject* item;
  PyObject* pylong;
  Py_ssize_t len = PySequence_Length((PyObject*)pyobj);
  mxArray* rowvec = mxCreateNumericMatrix(1,len,mxINT64_CLASS, mxREAL);
  long long* row = (long long*) mxGetData(rowvec);
  mwIndex i;
  for (i=0; i<len; i++) {
    item = PySequence_GetItem((PyObject*) pyobj, i);
    pylong = PyNumber_Long(item);
    row[i] = PyLong_AsLongLong(pylong);
    Py_DECREF(pylong);
    Py_DECREF(item);
  }
  return rowvec;
}


PRIVATE PyObject* Any_mxArray_to_PyObject(const mxArray* mxobj) {
  if (mxIsPyObject(mxobj)) {
    PyObject* pyobj = unbox(mxobj);
    Py_XINCREF(pyobj);
    return pyobj;
  }
  else if (mxIsChar(mxobj)) {
    return mxChar_to_PyString(mxobj);
  }
  else if ((mxIsNumeric(mxobj) || mxIsLogical(mxobj)) && 
	   mxGetNumberOfElements(mxobj) <= 1) {
    if (mxIsEmpty(mxobj))
      Py_RETURN_NONE;
    else
      return mxNumber_to_PyObject(mxobj, 0);
  }
  else if (mxIsCell(mxobj)) {
    return mxCell_to_PyTuple(mxobj);
  }
  else if (mxIsNumeric(mxobj) || mxIsLogical(mxobj))
    return mxNonScalar_to_PyList(mxobj); 
  else {
    mexErrMsgIdAndTxt("pymex:convertMXtoPY", "Don't know how to convert %s",
		      mxGetClassName(mxobj));
  }
}

PRIVATE PyObject* PyDict_from_KW(const mxArray* kwargs) {
  PyObject* dict = PyDict_New();
  mwSize numkeys = mxGetNumberOfElements(kwargs);
  mwIndex i;
  for (i=0; i<numkeys; i++) {
    PyObject* key = Any_mxArray_to_PyObject(mxGetProperty(kwargs, i, "keyword"));
    PyObject* val = Any_mxArray_to_PyObject(mxGetProperty(kwargs, i, "value"));
    PyDict_SetItem(dict, key, val);
  }
  return dict;
}
