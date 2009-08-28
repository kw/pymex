/* format is:
PYMEX(NAME, minimum_number_of_args, maximum_number_of_args, { function body} )
*/

PYMEX(MEXLOCK, 0,0, {
    mexLock();
  })

PYMEX(MEXUNLOCK, 0,0, {
    mexUnlock();
  })

PYMEX(MEXISLOCKED, 0,0, {
    plhs[0] = mxCreateLogicalScalar(mexIsLocked());
  })

PYMEX(DELETE_OBJ, 1,1, {
    if (!mxIsPyNull(prhs[0])) {
      PyObject* pyobj = unbox(prhs[0]);
      Py_DECREF(pyobj);
      mexUnlock();
    }
  })

PYMEX(GET_BUILTINS, 0,0, {
    plhs[0] = boxb(PyEval_GetBuiltins());
  })

PYMEX(IMPORT, 1,1, {
    if (!mxIsChar(prhs[0]))
      mexErrMsgTxt("import argument not string.");
    PyObject* name = mxChar_to_PyString(prhs[0]);
    PyObject* pyobj = PyImport_Import(name);
    Py_DECREF(name);
    plhs[0] = box(pyobj);
  })

PYMEX(IS, 2,2, {
    if (!mxIsPyObject(prhs[0]) || !mxIsPyObject(prhs[1]))
      plhs[0] = mxCreateLogicalScalar(0);
    else
      plhs[0] = mxCreateLogicalScalar(unbox(prhs[0]) == unbox(prhs[1]));
  })

PYMEX(TO_BOOL, 1,1, {
    plhs[0] = mxCreateLogicalScalar(PyObject_IsTrue(unbox(prhs[0])));
  })


#define PYMEX_BIN_OP(name, pyfun)			\
  PYMEX(name, 2,2, {					\
      PyObject* L = unboxn(prhs[0]);			\
      PyObject* R = unboxn(prhs[1]);			\
      plhs[0] = box(pyfun(L,R));			\
      Py_XDECREF(L);					\
      Py_XDECREF(R);					\
    })

PYMEX_BIN_OP(ADD, PyNumber_Add)
PYMEX_BIN_OP(SUBTRACT, PyNumber_Subtract)
PYMEX_BIN_OP(MULTIPLY, PyNumber_Multiply)
PYMEX_BIN_OP(DIVIDE, PyNumber_Divide)
PYMEX_BIN_OP(REM, PyNumber_Remainder)
PYMEX_BIN_OP(MOD, PyNumber_Divmod)
PYMEX_BIN_OP(BITAND, PyNumber_And)
PYMEX_BIN_OP(BITOR, PyNumber_Or)
PYMEX_BIN_OP(BITXOR, PyNumber_Xor)
#undef PYMEX_BIN_OP



PYMEX(TO_STR, 1,1, {
    if (!mxIsPyObject(prhs[0]))
      mexErrMsgTxt("argument must be a boxed pyobject");
    plhs[0] = PyObject_to_mxChar(unbox(prhs[0]));
  })

PYMEX(DIR, 1,1, {
    PyObject* pyobj;
    pyobj = PyObject_Dir(unbox(prhs[0]));
    plhs[0] = box(pyobj);
  })

PYMEX(GET_ATTR, 2,2, {
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

PYMEX(GET_TYPE, 1,1, {
    PyObject* pyobj = unbox(prhs[0]);
    plhs[0] = box(PyObject_Type(pyobj));
  })

PYMEX(SCALAR_TO_PYOBJ, 1,1, {
    plhs[0] = box(Any_mxArray_to_PyObject(prhs[0], PREFER_SCALAR));
  })

PYMEX(TO_LIST, 1,1, {
    plhs[0] = box(Any_mxArray_to_PyObject(prhs[0], PREFER_LIST));
  })

PYMEX(TO_TUPLE, 1,1, {
    plhs[0] = box(Any_mxArray_to_PyObject(prhs[0], PREFER_TUPLE));
  })

PYMEX(CALL, 2,3, {
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
  })

PYMEX(DICT_FROM_KW, 1,1, {
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

PYMEX(GET_ITEM, 1,1, {
    PyObject* pyobj = unbox(prhs[0]);
    PyObject* key = Any_mxArray_to_PyObject(prhs[1], PREFER_SCALAR);
    plhs[0] = box(PyObject_GetItem(pyobj, key));
  })

PYMEX(IS_CALLABLE, 1,1, {
    plhs[0] = mxCreateLogicalScalar(PyCallable_Check(unbox(prhs[0])));
  })

PYMEX(SET_ATTR, 3,3, {
    PyObject* pyobj = unbox(prhs[0]);
    PyObject* key = Any_mxArray_to_PyObject(prhs[1], PREFER_SCALAR);
    PyObject* val = Any_mxArray_to_PyObject(prhs[2], PREFER_SCALAR);
    PyObject_SetAttr(pyobj, key, val);
  })

PYMEX(SET_ITEM, 3,3, {
    PyObject* pyobj = unbox(prhs[0]);
    PyObject* key = Any_mxArray_to_PyObject(prhs[1], PREFER_SCALAR);
    PyObject* val = Any_mxArray_to_PyObject(prhs[2], PREFER_SCALAR);
    PyObject_SetItem(pyobj, key, val);
  })

PYMEX(GET_MODULE_DICT, 0,0, {
    plhs[0] = boxb(PyImport_GetModuleDict());
  })

PYMEX(IS_INSTANCE, 2,2, {
    plhs[0] = 
      mxCreateLogicalScalar(PyObject_IsInstance(unbox(prhs[0]),
						Any_mxArray_to_PyObject(prhs[1], 
									PREFER_SCALAR | PREFER_TUPLE)));
  })
