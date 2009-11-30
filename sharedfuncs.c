/* Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
   For full license details, see the LICENSE file. */

#include "pymex.h"
#include <mex.h>

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
/* 512 is probably a bit too generous. I believe MATLAB has a builtin limit - what is it? */
#define MAX_MXTYPE_NAME_SIZE 512
mxArray *box_by_type(PyObject *pyobj) {
  static char mlname[MAX_MXTYPE_NAME_SIZE] = {0};
  static char *package = "py.types.%s.%s";
  PYMEX_DEBUG("Trying to box %p\n", pyobj);
  mxArray *box = NULL;
  mxArray *mxname;
  mxArray *which;
  mxArray *err = NULL;
  if (!pyobj) {
    err = mexCallMATLABWithTrap(1,&box,0,NULL,PYMEX_MATLAB_VOIDPTR);
  }
  else {
    PyObject *type = (PyObject *) pyobj->ob_type;
    PyObject *mro;
    if (PyType_Check(pyobj)) {
      /* Calling type on a type gives us back type, and
	 calling mro on type doesn't work quite the same,
	 so we special case this one. We don't actually have
	 a wrapper type to wrap type, but if we did, it would.
	 Yo dawg.
      */
      PYMEX_DEBUG("Object is a type...\n");
      mro = PyTuple_Pack(1, &PyType_Type);
    }
    else {
      PYMEX_DEBUG("Object is not a type...\n");
      mro = PyObject_CallMethod(type, "mro", "()");
    }
    Py_ssize_t len = PySequence_Length(mro);
    Py_ssize_t i;
    for (i=0; i<len; i++) {
      PyObject *item = PySequence_GetItem(mro, i);
      if (item == pyobj) {
	PYMEX_DEBUG("Pointers match!?\n"); 
      }
      else if (!item) {
	PYMEX_DEBUG("Item is null!?\n");
      }
      else {
	PYMEX_DEBUG("Ok, getting name...\n");
      }
      PyObject *modname = PyObject_GetAttrString(item, "__module__");
      PyObject *cleanmodname = PyObject_CallMethod(modname, "strip", "s", "_");
      PyObject *name = PyObject_GetAttrString(item, "__name__");
      snprintf(mlname, MAX_MXTYPE_NAME_SIZE, package, 
	       PyBytes_AsString(cleanmodname), PyBytes_AsString(name));
      PYMEX_DEBUG("Checking for %s...\n", mlname);
      Py_DECREF(name);
      Py_DECREF(cleanmodname);
      Py_DECREF(modname);
      Py_DECREF(item);
      mxname = mxCreateString(mlname);
      mxArray *werr = mexCallMATLABWithTrap(1,&which,1,&mxname,"which");
      mxDestroyArray(mxname);
      if (!werr && mxGetNumberOfElements(which) > 0) {
	PYMEX_DEBUG("%s looks good, trying it...\n", mlname);
	err = mexCallMATLABWithTrap(1,&box,0,NULL,mlname);
	mxDestroyArray(which);
	if (!err) break;
      }
      else {
	mxDestroyArray(which);
      }
    }
    Py_DECREF(mro);
    if (err || !box) { /* none found, use sane default */
      PYMEX_DEBUG("No reasonable box found, using default.\n");
      err = mexCallMATLABWithTrap(1,&box,0,NULL,PYMEX_MATLAB_PYOBJECT);
    }
  }
  if (err || !box) {
    PYMEX_DEBUG("Unable to find " PYMEX_MATLAB_PYOBJECT);
    PyErr_Format(MATLABError,"Unable to find %s", PYMEX_MATLAB_PYOBJECT);
    return NULL;
  }
  PYMEX_DEBUG("Returning box\n");
  return box;
}

/* boxes the object, stealing the reference */
mxArray *box (PyObject *pyobj) {
  mxArray *boxed = NULL;
  if (!pyobj) {
    PYMEX_DEBUG("Boxing null object.");
  }
  boxed = box_by_type(pyobj);
  if (!boxed) return NULL;
  if (pyobj) mexLock();
  mxArray *ptr_field = mxGetProperty(boxed, 0, "pointer");
  void **ptr = mxGetData(ptr_field);
  *ptr = (void*) pyobj;
  mxSetProperty(boxed, 0, "pointer", ptr_field);
  return boxed;
}

/* Box a borrowed reference (increfs it first) */
mxArray *boxb (PyObject *pyobj) {
  Py_XINCREF(pyobj);
  return box(pyobj);
}

/* Unboxes an object, returning a borrowed reference */
PyObject *unbox (const mxArray *mxobj) {  
  if (!mxobj) return PyErr_Format(MATLABError, "Can't unbox from null pointer");
  if (mxIsPyNull(mxobj)) {
    PYMEX_DEBUG("Unboxed a null object.");
    return PyErr_Format(MATLABError, "Unboxed pointer is null.");
  }
  else {
    void **ptr = mxGetData(mxGetProperty(mxobj, 0, "pointer"));
    return (PyObject *) *ptr;
  }
}

/* Unboxes an object, returning a new reference */
PyObject *unboxn (const mxArray *mxobj) {
  PyObject *pyobj;
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
bool mxIsPyNull (const mxArray *mxobj) {
  void **ptr = mxGetData(mxGetProperty(mxobj, 0, "pointer"));
  return !*ptr;
}

/* Asks the MATLAB interpreter if the object isa subclass of voidptr.
   This includes null pointers and things not descended from object.
 */
bool mxIsPyObject(const mxArray *mxobj) {
  mxArray *boolobj;
  mxArray *args[2];
  args[0] = (mxArray *) mxobj;
  args[1] = mxCreateString(PYMEX_MATLAB_VOIDPTR);
  mexCallMATLABWithTrap(1,&boolobj,2,args,"isa");
  mxDestroyArray(args[1]);
  return mxIsLogicalScalarTrue(boolobj);
}

PyObject *mxChar_to_PyBytes(const mxArray *mxchar) {
  if (!mxchar || !mxIsChar(mxchar))
    mexErrMsgTxt("Input isn't a mxChar");
  char *tempstring = mxArrayToString(mxchar);
  if (!tempstring)
    mexErrMsgTxt("Couldn't stringify mxArray for some reason.");
  PyObject *pystr = PyBytes_FromString(tempstring);
  mxFree(tempstring);
  if (!pystr)
    mexErrMsgTxt("Couldn't convert from string to PyBytes");
  return pystr;
}

mxArray *PyBytes_to_mxChar(PyObject *pystr) {
  if (!pystr || !PyBytes_Check(pystr))
    mexErrMsgTxt("Input isn't a PyBytes");
  char *tempstring = PyBytes_AsString( pystr);
  mxArray *mxchar = mxCreateString(tempstring);
  return mxchar;
}

mxArray *PyObject_to_mxChar(PyObject *pyobj) {
  if (pyobj) {
    PyObject *pystr = PyObject_Str(pyobj);
    mxArray *mxchar = PyBytes_to_mxChar(pystr);
    Py_DECREF(pystr);
    return mxchar;
  } 
  else {
    PyErr_Format(MATLABError, "Can't convert NULL to string");
    return NULL;
  }
}

PyObject *mxCell_to_PyTuple(const mxArray *mxobj) {
  mwSize numel = mxGetNumberOfElements(mxobj);
  PyObject *pyobj = PyTuple_New(numel);
  mwSize i;
  for (i=0; i<numel; i++) {
    PyTuple_SetItem(pyobj, i, Any_mxArray_to_PyObject(mxGetCell(mxobj, i)));
  }
  return pyobj;
}

PyObject *mxCell_to_PyTuple_recursive(const mxArray *mxobj) {
  mwSize numel = mxGetNumberOfElements(mxobj);
  PyObject *pyobj = PyTuple_New(numel);
  mwSize i;
  for (i=0; i<numel; i++) {
    mxArray *item = mxGetCell(mxobj, i);
    PyObject *pyitem;
    if (mxIsCell(item))
      pyitem = mxCell_to_PyTuple_recursive(item);
    else
      pyitem = Any_mxArray_to_PyObject(item);
    PyTuple_SetItem(pyobj, i, pyitem);
  }
  return pyobj;
}

mxArray *PySequence_to_mxCell(PyObject *pyobj) {
  PyObject *item;
  Py_ssize_t len = PySequence_Length(pyobj);
  mxArray *mxcell = mxCreateCellMatrix(1,len);
  PERSIST_ARRAY(mxcell);
  mwIndex i;
  for (i=0; i<len; i++) {
    item = PySequence_GetItem(pyobj, i);
    mxSetCell(mxcell, i, Any_PyObject_to_mxArray(item));
    Py_DECREF(item);
  }
  return mxcell;
}

PyObject *Any_mxArray_to_PyObject(const mxArray *mxobj) {
  if (mxIsPyObject(mxobj)) {
    PyObject *pyobj = unbox(mxobj);
    Py_XINCREF(pyobj);
    return pyobj;
  }
  else if (mxIsChar(mxobj)) {
    return mxChar_to_PyBytes(mxobj);
  }
  else {
    return Py_mxArray_New((mxArray *) mxobj,1);
  }
}

mxArray *Any_PyObject_to_mxArray(PyObject *pyobj) {
  if (!pyobj)
    return box(pyobj); /* Null pointer */
  if (Py_mxArray_Check(pyobj))
    return mxArrayPtr(pyobj);
  else {
    PyObject *utils = PyImport_ImportModule("pymexutil");
    PyObject *arg;
    if (PyTuple_Check(pyobj)) arg = PyTuple_Pack(1, pyobj);
    else {
      arg = pyobj;
      Py_INCREF(pyobj);
    }
    PyObject *unpyed = PyObject_CallMethod(utils, "unpy", "O", arg);
    Py_DECREF(arg);
    if (PyErr_Occurred()) {
      if (PyErr_ExceptionMatches(PyExc_NotImplementedError))
	return boxb(pyobj);
      else return NULL;
    }
    if (Py_mxArray_Check(unpyed)) {
      mxArray *retval = mxDuplicateArray(mxArrayPtr(unpyed));
      PERSIST_ARRAY(retval);
      Py_DECREF(unpyed);
      return retval;
    }
    else return boxb(pyobj);
  }
}

/* TODO: Low priority, but this depends on libmex. If we try to make an
   external module later, this will need to function to some extent.
   It could operate at low capability by using a hardcoded list for
   the builtins and assuming that anything else has _object as a super.
 */
PyObject *Calculate_matlab_mro(mxArray *mxobj) {
  mxArray *argin[3];
  argin[0] = mxobj;
  argin[1] = mxCreateLogicalScalar(1); /* addvirtual=true */
  argin[2] = mxCreateLogicalScalar(1); /* autosplit=true */
  mxArray *argout[1] = {NULL};
  mxArray *err = mexCallMATLABWithTrap(1, argout, 3, argin, "mro");
  if (err) return PyObject_CallMethod(mexmodule, "__raiselasterror", "()");
  else {
    PyObject *retval = mxCell_to_PyTuple_recursive(argout[0]);
    mxDestroyArray(argout[0]);
    mxDestroyArray(argin[1]);
    mxDestroyArray(argin[2]);
    return retval;
  }
}

/* Attempts to locate an appropriate subclass of mx.Array 
   using the mltypes package. If for some reason this fails, 
   mx.Array is returned instead.
 */
PyObject *Find_mltype_for(mxArray *mxobj) {
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


PyObject *Py_mxArray_New(mxArray *mxobj, bool duplicate) {
  mxArray *copy;
  if (duplicate) {
    copy = mxDuplicateArray(mxobj);
    mexMakeArrayPersistent(copy);
  }
  else {
    copy = mxobj;
  }
  PyObject *mxptr = mxArrayPtr_New(copy);
  PyObject *args = PyTuple_New(0);
  PyObject *kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "mxpointer", mxptr);
  PyObject *arraycls = Find_mltype_for(copy);
  /* TODO: There is probably a better way to do this... */
  PyObject *ret = PyObject_Call(arraycls, args, kwargs);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_DECREF(mxptr);
  Py_DECREF(arraycls);
  return ret;
}

int Py_mxArray_Check(PyObject *pyobj) {
  return PyObject_IsInstance(pyobj, PyObject_GetAttrString(mxmodule, "Array"));
}

mxArray *PyObject_to_mxDouble(PyObject *pyobj) {
  double val = PyFloat_AsDouble(pyobj);
  if (PyErr_Occurred()) return NULL;
  mxArray *mxval = mxCreateDoubleScalar(val);
  PERSIST_ARRAY(mxval);
  return mxval;
}

mxArray *PyObject_to_mxLong(PyObject *pyobj) {
  long val = PyLong_AsLong(pyobj);
  if (PyErr_Occurred()) return NULL;
  /* there is no mxCreateNumericScalar */
  mxArray *mxval = mxCreateNumericMatrix(1, 1, mxINT64_CLASS, mxREAL);
  PERSIST_ARRAY(mxval);
  long* ptr = (long*) mxGetData(mxval);
  ptr[0] = val;
  return mxval;
}

mxArray *PyObject_to_mxLogical(PyObject *pyobj) {
  return mxCreateLogicalScalar(PyObject_IsTrue(pyobj));
}

char mxClassID_to_Numpy_Typekind(mxClassID mxclass) {
  switch (mxclass) {
  case mxLOGICAL_CLASS: return 'b';
  case mxCHAR_CLASS: return 'S';
  case mxINT8_CLASS: 
  case mxINT16_CLASS: 
  case mxINT32_CLASS: 
  case mxINT64_CLASS: return 'i'; 
  case mxUINT8_CLASS: 
  case mxUINT16_CLASS: 
  case mxUINT32_CLASS: 
  case mxUINT64_CLASS: return 'u'; 
  case mxSINGLE_CLASS: 
  case mxDOUBLE_CLASS: return 'f';
  default: return 'V';
  }
}

mxArray *mxArrayPtr(PyObject *pyobj) {
  PyObject *ptr;
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
  return (mxArray *) PyCObject_AsVoidPtr(ptr);
}

static void _mxArrayPtr_destructor(void *mxobj, void *desc) {
  Py_XDECREF((PyObject *) desc);  
  if (mxobj) mxDestroyArray((mxArray *) mxobj);
}

PyObject *mxArrayPtr_New(mxArray *mxobj) {
  if (!mxmodule)
    return PyErr_Format(PyExc_RuntimeError, "mxmodule not yet initialized");
  PERSIST_ARRAY(mxobj);
  Py_INCREF(mxmodule);
  return PyCObject_FromVoidPtrAndDesc(mxobj, mxmodule, _mxArrayPtr_destructor);
}

int mxArrayPtr_Check(PyObject *obj) {
  return (obj && PyCObject_Check(obj) && PyCObject_GetDesc(obj) == mxmodule);
}
