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
    plhs[0] = PyObject_to_mxLogical(unbox(prhs[0]));
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
PYMEX_BIN_OP(LSHIFT, PyNumber_Lshift)
PYMEX_BIN_OP(RSHIFT, PyNumber_Rshift)
#undef PYMEX_BIN_OP

#define PYMEX_UNARY_OP(name, pyfun)		\
  PYMEX(name, 1,1, {				\
      PyObject* O = unboxn(prhs[0]);		\
      plhs[0] = box(pyfun(O));			\
      Py_XDECREF(O);				\
    })

PYMEX_UNARY_OP(NEGATE, PyNumber_Negative)
PYMEX_UNARY_OP(POSIFY, PyNumber_Positive)
PYMEX_UNARY_OP(ABS, PyNumber_Absolute)
PYMEX_UNARY_OP(INVERT, PyNumber_Invert)
#undef PYMEX_UNARY_OP

PYMEX(POWER, 2,3, {
    PyObject* x = unboxn(prhs[0]);
    PyObject* y = unboxn(prhs[1]);
    PyObject* z;
    if (nrhs != 3) {
      z = Py_None;
      Py_INCREF(z);
    }
    else {
      z = unboxn(prhs[2]);
    }
    plhs[0] = box(PyNumber_Power(x, y, z));
    Py_XDECREF(x);
    Py_XDECREF(y);
    Py_XDECREF(z);
  })

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


PYMEX(GET_TYPE, 1,1, {
    PyObject* pyobj = unbox(prhs[0]);
    plhs[0] = box(PyObject_Type(pyobj));
  })

PYMEX(SCALAR_TO_PYOBJ, 1,1, {
    plhs[0] = box(Any_mxArray_to_PyObject(prhs[0]));
  })

PYMEX(TO_LIST, 1,1, {
    plhs[0] = box(Any_mxArray_to_PyObject(prhs[0]));
  })

PYMEX(TO_TUPLE, 1,1, {
    plhs[0] = box(Any_mxArray_to_PyObject(prhs[0]));
  })

PYMEX(AS_DOUBLE,1,1, {
    plhs[0] = PyObject_to_mxDouble(unbox(prhs[0]));
  })

PYMEX(AS_LONG,1,1, {
    plhs[0] = PyObject_to_mxLong(unbox(prhs[0]));
  })

PYMEX(CALL, 2,3, {
    PyObject* callobj = unbox(prhs[0]);
    if (!PyCallable_Check(callobj))
      mexErrMsgIdAndTxt("python:NotCallable", "tried to call object which is not callable.");
    PyObject* args = mxCell_to_PyTuple(prhs[1]);
    if (!PyTuple_Check(args))
      mexErrMsgIdAndTxt("python:NotTuple", "args must be a tuple");    
    PyObject* kwargs = NULL;
    if (nrhs > 2) {
      kwargs = PyDict_from_KW(prhs[2]);
      if (kwargs && !PyDict_Check(kwargs))
	mexErrMsgIdAndTxt("python:NoKWargs", "kwargs must be a dict or null");
    }
    PyObject* result = PyObject_Call(callobj, args, kwargs);
    plhs[0] = box(result);
  })

PYMEX(IS_CALLABLE, 1,1, {
    plhs[0] = mxCreateLogicalScalar(PyCallable_Check(unbox(prhs[0])));
  })

PYMEX(GET_ATTR, 2,2, {
    PyObject* pyobj = unbox(prhs[0]);
    PyObject* name = unboxn(prhs[1]);
    plhs[0] = box(PyObject_GetAttr(pyobj, name));
    Py_XDECREF(name);
  })

PYMEX(SET_ATTR, 3,3, {
    PyObject* pyobj = unbox(prhs[0]);
    PyObject* key = unboxn(prhs[1]);
    PyObject* val = unboxn(prhs[2]);
    PyObject_SetAttr(pyobj, key, val);
    Py_XDECREF(key);
    Py_XDECREF(val);
  })

PYMEX(HAS_ATTR, 2, 2, {
    PyObject* pyobj = unbox(prhs[0]);
    PyObject* name = unboxn(prhs[1]);
    plhs[0] = mxCreateLogicalScalar(PyObject_HasAttr(pyobj, name));
    Py_XDECREF(name);
  })

PYMEX(GET_ITEM, 2,2, {
    PyObject* pyobj = unbox(prhs[0]);
    PyObject* key = Any_mxArray_to_PyObject(prhs[1]);
    plhs[0] = box(PyObject_GetItem(pyobj, key));
  })

PYMEX(SET_ITEM, 3,3, {
    PyObject* pyobj = unbox(prhs[0]);
    PyObject* key = Any_mxArray_to_PyObject(prhs[1]);
    PyObject* val = Any_mxArray_to_PyObject(prhs[2]);
    PyObject_SetItem(pyobj, key, val);
  })

PYMEX(GET_MODULE_DICT, 0,0, {
    plhs[0] = boxb(PyImport_GetModuleDict());
  })

PYMEX(IS_INSTANCE, 2,2, {
    plhs[0] = 
      mxCreateLogicalScalar(PyObject_IsInstance(unbox(prhs[0]),
						Any_mxArray_to_PyObject(prhs[1])));
  })

PYMEX(TO_MXARRAY, 1,1, {
    if (!mxIsPyObject(prhs[0]))
      plhs[0] = (mxArray*) prhs[0];
    else
      plhs[0] = Any_PyObject_to_mxArray(unbox(prhs[0]));
  })

PYMEX(RUN_SIMPLE_STRING, 1, 1, {
    char* cmd = mxArrayToString(prhs[0]);
    int ret = PyRun_SimpleString(cmd);
    mxFree(cmd);
    plhs[0] = mxCreateDoubleScalar((double) ret);
  })

PYMEX(AS_PYCOBJECT, 1, 1, {
    plhs[0] = box(PyCObject_from_mxArray(prhs[0]));
  })
		  
