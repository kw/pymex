typedef enum {
  PREFER_TUPLE = 1,
  PREFER_LIST = 2,
  PREFER_SCALAR = 4,
} prefer_pytype;


static mxArray* box(const PyObject* pyobj);
static mxArray* boxb(const PyObject* pyobj);
static PyObject* unbox (const mxArray* mxobj);
static PyObject* unboxn (const mxArray* mxobj);
static bool mxIsPyNull (const mxArray* mxobj);
static bool mxIsPyObject(const mxArray* mxobj);
static PyObject* mxChar_to_PyString(const mxArray* mxchar);
static mxArray* PyString_to_mxChar(const PyObject* pystr);
static mxArray* PyObject_to_mxChar(const PyObject* pyobj);
static PyObject* mxCell_to_PyObject(const mxArray* mxobj, prefer_pytype pref);
static PyObject* mxVector_to_PyObject(const mxArray* mxobj, prefer_pytype pref);
static PyObject* mxNumber_to_PyObject(const mxArray* mxobj, mwIndex ind);
static PyObject* Any_mxArray_to_PyObject(const mxArray* mxobj, prefer_pytype pref);


static mxArray* box (const PyObject* pyobj) {
  mxArray* boxed = NULL;
  mexCallMATLAB(1,&boxed,0,NULL,"py.Object");
  if (!pyobj) {
    PYMEX_DEBUG("Attempted to box null object.");
  } else {
    mexLock();
    mxArray* ptr_field = mxGetProperty(boxed, 0, "pointer");
    UINT64* ptr = mxGetData(ptr_field);
    *ptr = (UINT64) pyobj;
    mxSetProperty(boxed, 0, "pointer", ptr_field);
  }
  return boxed;
}

static mxArray* boxb (const PyObject* pyobj) {
  Py_XINCREF(pyobj);
  return box(pyobj);
}

static PyObject* unbox (const mxArray* mxobj) {  
  if (mxIsPyNull(mxobj)) {
    PYMEX_DEBUG("Attempt to unbox null object.");
    return NULL;
  }
  else {
    UINT64* ptr = mxGetData(mxGetProperty(mxobj, 0, "pointer"));
    return (PyObject*) *ptr;
  }
}

static PyObject* unboxn (const mxArray* mxobj) {
  PyObject* pyobj;
  if (mxIsPyObject(mxobj)) {
    pyobj = unbox(mxobj);
    Py_XINCREF(pyobj);
  }
  else {
    pyobj = Any_mxArray_to_PyObject(mxobj, PREFER_SCALAR);
  }
  return pyobj;
}

static bool mxIsPyNull (const mxArray* mxobj) {
  UINT64* ptr = mxGetData(mxGetProperty(mxobj, 0, "pointer"));
  return !*ptr;
}

static bool mxIsPyObject(const mxArray* mxobj) {
  return mxIsClass(mxobj, "py.Object");
}

static PyObject* mxChar_to_PyString(const mxArray* mxchar) {
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

static mxArray* PyString_to_mxChar(const PyObject* pystr) {
  if (!pystr || !PyString_Check(pystr))
    mexErrMsgTxt("Input isn't a PyString");
  char* tempstring = PyString_AsString((PyObject*) pystr);
  mxArray* mxchar = mxCreateString(tempstring);
  return mxchar;
}

static mxArray* PyObject_to_mxChar(const PyObject* pyobj) {
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

/*static PyObject* mxCell_to_PyObject(const mxArray* mxobj, prefer_pytype pref) {
  mwSize numel = mxGetNumberOfElements(mxobj);
  PyObject* pyobj = (pref & PREFER_LIST) ? PyList_New(numel) : PyTuple_New(numel);
  mwSize i;
  for (i=0; i<numel; i++) {
    PyObject* val = 
    if (pref & PREFER_LIST)
      PyList_SetItem(pyobj, i, val);
    else
      PyTuple_SetIteem(pyobj, i, val);
  }
  return pyobj;
  }*/

static PyObject* mxNonScalar_to_PySequence(const mxArray* mxobj, prefer_pytype pref) {
  mwSize numel = mxGetNumberOfElements(mxobj);
  PyObject* pyobj = (pref & PREFER_LIST) ? PyList_New(numel) : PyTuple_New(numel);
  mwSize i;
  mxClassID type = mxGetClassID(mxobj);
  PyObject* val;
  for (i=0; i<numel; i++) {
    switch (type) {
    case mxCELL_CLASS:
      val = Any_mxArray_to_PyObject(mxGetCell(mxobj, i), pref); break;
    default:
      val = mxNumber_to_PyObject(mxobj, i); break;
    }
    if (pref & PREFER_LIST)
      PyList_SetItem(pyobj, i, val);
    else
      PyTuple_SetItem(pyobj, i, val);
  }
  return pyobj;
}


/*
static PyObject* mxCell_to_PyList(const mxArray* mxobj, prefer_pytype pref) {
  mwSize numel = mxGetNumberOfElements(mxobj);
  PyObject* pyobj = PyList_New(numel);
  mwSize i;
  for (i=0; i<numel; i++) {
    PyList_SetItem(pyobj, i, Any_mxArray_to_PyObject(mxGetCell(mxobj, i), pref));
  }
  return pyobj;
}

static PyObject* mxCell_to_PyTuple(const mxArray* mxobj, prefer_pytype pref) {
  mwSize numel = mxGetNumberOfElements(mxobj);
  PyObject* pyobj = PyTuple_New(numel);
  mwSize i;
  for (i=0; i<numel; i++) {
    PyTuple_SetItem(pyobj, i, Any_mxArray_to_PyObject(mxGetCell(mxobj, i), pref));
  }
  return pyobj;
}
*/

static PyObject* mxNumber_to_PyObject(const mxArray* mxobj, mwIndex index) {
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
    

static PyObject* Any_mxArray_to_PyObject(const mxArray* mxobj, prefer_pytype pref) {
  if (mxIsPyObject(mxobj)) {
    PyObject* pyobj = unbox(mxobj);
    Py_XINCREF(pyobj);
    return pyobj;
  }
  else if (mxIsChar(mxobj)) {
    return mxChar_to_PyString(mxobj);
  }
  else if ((mxIsNumeric(mxobj) || mxIsLogical(mxobj)) && 
	   mxGetNumberOfElements(mxobj) <= 1 &&
	   pref & PREFER_SCALAR) {
    if (mxIsEmpty(mxobj))
      Py_RETURN_NONE;
    else
      return mxNumber_to_PyObject(mxobj, 0);
  }
  else if (mxIsNumeric(mxobj) || mxIsLogical(mxobj) || mxIsCell(mxobj))
    return mxNonScalar_to_PySequence(mxobj, pref); 
  else {
    mexErrMsgIdAndTxt("pymex:convertMXtoPY", "Don't know how to convert %s",
		      mxGetClassName(mxobj));
  }
}     
	
	
