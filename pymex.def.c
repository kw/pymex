
PYMEX(MEXLOCK, 1, {
    mexLock();
  })

PYMEX(MEXUNLOCK, 2, {
    mexUnlock();
  })

PYMEX(DELETE_OBJ, 3, {
    if (nrhs < 1)
      mexErrMsgTxt("Argument to DELETE_OBJ must be a boxed PyObject");
    if (!mxIsPyNull(prhs[0])) {
      PyObject* pyobj = unbox(prhs[0]);
      Py_DECREF(pyobj);
      mexUnlock();
    }
  })

PYMEX(GET_BUILTINS, 6, {
    plhs[0] = boxb(PyEval_GetBuiltins());
  })

PYMEX(IMPORT, 8, {
    if (nrhs < 1 || !mxIsChar(prhs[0]))
      mexErrMsgTxt("import argument not string.");
    PyObject* name = mxChar_to_PyString(prhs[0]);
    PyObject* pyobj = PyImport_Import(name);
    Py_DECREF(name);
    plhs[0] = box(pyobj);
  })

PYMEX(TO_STR, 9, {
    if (nrhs < 1 || !mxIsPyObject(prhs[0]))
      mexErrMsgTxt("argument must be a boxed pyobject");
    plhs[0] = PyObject_to_mxChar(unbox(prhs[0]));
  })

PYMEX(DIR, 10, {
    PyObject* pyobj;
    pyobj = PyObject_Dir(unbox(prhs[0]));
    plhs[0] = box(pyobj);
  })

PYMEX(GET_ATTR, 11, {
    if (nrhs < 2)
      mexErrMsgTxt("must provide two arguments: object and attribute name");
    PyObject* pyobj = unbox(prhs[0]);
    PyObject* name;
    if (!mxIsPyObject(prhs[1]))
      name = mxChar_to_PyString(prhs[1]);
    else
      name = unbox(prhs[1]);
    plhs[0] = box(PyObject_GetAttr(pyobj, name));
    if (!mxIsPyObject(prhs[1]))
      Py_DECREF(name);
  })

PYMEX(GET_TYPE, 12, {
    if (nrhs < 1)
      mexErrMsgTxt("must provide one pyobj");
    PyObject* pyobj = unbox(prhs[0]);
    plhs[0] = box(PyObject_Type(pyobj));
  })

PYMEX(SCALAR_TO_PYOBJ, 13, {
    plhs[0] = box(Any_mxArray_to_PyObject(prhs[0], PREFER_SCALAR));
  })

PYMEX(TO_LIST, 14, {
    plhs[0] = box(Any_mxArray_to_PyObject(prhs[0], PREFER_LIST));
  })

PYMEX(TO_TUPLE, 15, {
    plhs[0] = box(Any_mxArray_to_PyObject(prhs[0], PREFER_TUPLE));
  })

PYMEX(CALL, 16, {
    PyObject* callobj = unbox(prhs[0]);
    if (!PyCallable_Check(callobj))
      mexErrMsgIdAndTxt("python:NotCallable", "tried to call object which is not callable.");
    PyObject* args = unbox(prhs[1]);
    if (!PyTuple_Check(args))
      mexErrMsgIdAndTxt("python:NotTuple", "args must be a tuple");    
    PyObject* kwargs = NULL;
    if (nrhs > 2) {
      kwargs = unbox(prhs[2]);
      if (kwargs && !PyDict_Check(kwargs))
	mexErrMsgIdAndTxt("python:NoKWargs", "kwargs must be a dict or null");
    }
    PyObject* result = PyObject_Call(callobj, args, kwargs);
    plhs[0] = box(result);
  });

PYMEX(DICT_FROM_KW, 17, {
    PyObject* dict = PyDict_New();
    /* Step 1: You put your dict in the box... */
    plhs[0] = box(dict); 
    mwSize numkeys = mxGetNumberOfElements(prhs[0]);
    mwSize i;
    for (i=0; i<numkeys; i++) {
      PyObject* key = Any_mxArray_to_PyObject(mxGetProperty(prhs[0], i, "keyword"), PREFER_SCALAR);
      PyObject* val = Any_mxArray_to_PyObject(mxGetProperty(prhs[0], i, "value"), PREFER_SCALAR);
      PyDict_SetItem(dict, key, val);
    }
  })

PYMEX(GET_ITEM, 18, {
    PyObject* pyobj = unbox(prhs[0]);
    PyObject* key = Any_mxArray_to_PyObject(prhs[1], PREFER_SCALAR);
    plhs[0] = box(PyObject_GetItem(pyobj, key));
  })

PYMEX(IS_CALLABLE, 19, {
    plhs[0] = mxCreateLogicalScalar(PyCallable_Check(unbox(prhs[0])));
  })

PYMEX(SET_ATTR, 20, {
    PyObject* pyobj = unbox(prhs[0]);
    PyObject* key = Any_mxArray_to_PyObject(prhs[1], PREFER_SCALAR);
    PyObject* val = Any_mxArray_to_PyObject(prhs[2], PREFER_SCALAR);
    PyObject_SetAttr(pyobj, key, val);
  })

PYMEX(SET_ITEM, 21, {
    PyObject* pyobj = unbox(prhs[0]);
    PyObject* key = Any_mxArray_to_PyObject(prhs[1], PREFER_SCALAR);
    PyObject* val = Any_mxArray_to_PyObject(prhs[2], PREFER_SCALAR);
    PyObject_SetItem(pyobj, key, val);
  })

PYMEX(GET_MODULE_DICT, 22, {
    plhs[0] = boxb(PyImport_GetModuleDict());
  })

PYMEX(IS_INSTANCE, 23, {
    plhs[0] = 
      mxCreateLogicalScalar(PyObject_IsInstance(unbox(prhs[0]),
						Any_mxArray_to_PyObject(prhs[1], 
									PREFER_SCALAR | PREFER_TUPLE)));
  })
