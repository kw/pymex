/* Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
   For full license details, see the LICENSE file. */

/* format is:
PYMEX(NAME, minimum_number_of_args, maximum_number_of_args, docstring, { function body} )
*/

PYMEX(MEXLOCK, 0,0, 
      "Adds one recursive lock from the mex file. The mex file can't be "
      "cleared from memory while a lock is held, and PYMEX makes use of this "
      "by adding one lock for each Python object handle. The destructor for "
      "the object handle automatically removes the lock using the DELETE_OBJ "
      "command rather than MEXUNLOCK.",
      {
	mexLock();
      })

PYMEX(MEXUNLOCK, 0,0, 
      "Opposite of MEXLOCK. Only removes a single lock.",
      {
	mexUnlock();
      })

PYMEX(MEXISLOCKED, 0,0, 
      "Determines whether any locks are held by this mex file.",
      {
	plhs[0] = mxCreateLogicalScalar(mexIsLocked());
      })

PYMEX(DELETE_OBJ, 1,1, 
      "Releases a reference to the given Python object and frees the mex lock "
      "associated with it. See MEXLOCK.",
      {
	if (!mxIsPyNull(prhs[0])) {
	  PyObject *pyobj = unbox(prhs[0]);
	  Py_DECREF(pyobj);
	  mexUnlock();
	}
      })

PYMEX(GET_BUILTINS, 0,0, 
      "Returns the Python builtins dictionary. "
      "Use the pybuiltins m-function to do this.",
      {
	plhs[0] = boxb(PyEval_GetBuiltins());
      })

PYMEX(IMPORT, 1,1, 
      "Attempts to import a Python module by name. "
      "Use the pyimport m-function to do this.",
      {
	if (!mxIsChar(prhs[0]))
	  mexErrMsgTxt("import argument not string.");
	PyObject *name = mxChar_to_PyBytes(prhs[0]);
	PyObject *pyobj = PyImport_Import(name);
	Py_DECREF(name);
	plhs[0] = box(pyobj);
      })

PYMEX(IS, 2,2, 
      "Given two python objects, determines whether they are the same "
      "object. This is a simple pointer comparison, and is equivalent "
      "to the python 'is' keyword.",
      {
	if (!mxIsPyObject(prhs[0]) || !mxIsPyObject(prhs[1]))
	  plhs[0] = mxCreateLogicalScalar(0);
	else
	  plhs[0] = mxCreateLogicalScalar(unbox(prhs[0]) == unbox(prhs[1]));
      })

PYMEX(TO_BOOL, 1,1, 
      "Convert a python object to a boolean value.",
      {
	plhs[0] = PyObject_to_mxLogical(unbox(prhs[0]));
      })

#define PYMEX_BIN_OP(name, pyfun)			\
  PYMEX(name, 2,2,					\
	"Binary operator: " #pyfun,			\
	{						\
	  PyObject *L = unboxn(prhs[0]);		\
	  PyObject *R = unboxn(prhs[1]);		\
	  plhs[0] = box(pyfun(L,R));			\
	  Py_XDECREF(L);				\
	  Py_XDECREF(R);				\
	})

PYMEX_BIN_OP(ADD, PyNumber_Add)
PYMEX_BIN_OP(SUBTRACT, PyNumber_Subtract)
PYMEX_BIN_OP(MULTIPLY, PyNumber_Multiply)
PYMEX_BIN_OP(DIVIDE, PyNumber_TrueDivide)
PYMEX_BIN_OP(REM, PyNumber_Remainder)
PYMEX_BIN_OP(MOD, PyNumber_Divmod)
PYMEX_BIN_OP(BITAND, PyNumber_And)
PYMEX_BIN_OP(BITOR, PyNumber_Or)
PYMEX_BIN_OP(BITXOR, PyNumber_Xor)
PYMEX_BIN_OP(LSHIFT, PyNumber_Lshift)
PYMEX_BIN_OP(RSHIFT, PyNumber_Rshift)
#undef PYMEX_BIN_OP

#define PYMEX_UNARY_OP(name, pyfun)		\
  PYMEX(name, 1,1,				\
	"Unary operator: " #pyfun,		\
	{					\
	  PyObject *O = unboxn(prhs[0]);	\
	  plhs[0] = box(pyfun(O));		\
	  Py_XDECREF(O);			\
	})

PYMEX_UNARY_OP(NEGATE, PyNumber_Negative)
PYMEX_UNARY_OP(POSIFY, PyNumber_Positive)
PYMEX_UNARY_OP(ABS, PyNumber_Absolute)
PYMEX_UNARY_OP(INVERT, PyNumber_Invert)
#undef PYMEX_UNARY_OP

#define PYMEX_CMP_OP(name)					\
  PYMEX(name, 2,2,						\
	"Comparison operator: " #name,				\
	{							\
	  PyObject *A = unboxn(prhs[0]);			\
	  PyObject *B = unboxn(prhs[1]);			\
	  plhs[0] = box(PyObject_RichCompare(A, B, Py_##name));	\
	  Py_XDECREF(A);					\
	  Py_XDECREF(B);					\
	})

PYMEX_CMP_OP(LT)
PYMEX_CMP_OP(LE)
PYMEX_CMP_OP(EQ)
PYMEX_CMP_OP(GT)
PYMEX_CMP_OP(GE)
PYMEX_CMP_OP(NE)

#undef PYMEX_CMP_OP

PYMEX(POWER, 2,3,
      "Python's power operator. Has an optional third argument, "
      "see the python docs for details. ",
      {
	PyObject *x = unboxn(prhs[0]);
	PyObject *y = unboxn(prhs[1]);
	PyObject *z;
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

PYMEX(TO_STR, 1,1,
      "Attempt to produce a string representation of a "
      "python object.",
      {
	if (!mxIsPyObject(prhs[0]))
	  mexErrMsgTxt("argument must be a boxed pyobject");
	plhs[0] = PyObject_to_mxChar(unbox(prhs[0]));
      })

PYMEX(DIR, 1,1, 
      "Retrieves the object's attribute directory.",
      {
	PyObject *pyobj;
	pyobj = PyObject_Dir(unbox(prhs[0]));
	plhs[0] = box(pyobj);
      })


PYMEX(GET_TYPE, 1,1, 
      "Retrieves the object's type.",
      {
	PyObject *pyobj = unbox(prhs[0]);
	plhs[0] = box(PyObject_Type(pyobj));
      })

PYMEX(TO_PYOBJECT, 1,1, 
      "Coerce a MATLAB object to a Python type",
      {    
	plhs[0] = box(Any_mxArray_to_PyObject(prhs[0]));
      })

#if PYMEX_DEBUG_FLAG
PYMEX(CALL, 2,3, 
      "Calls a callable python object. In addition to the "
      "object itself, the second argument is a cell array or tuple "
      "of arguments. An optional third argument is a dict of keyword arguments. "
      "No output unpacking is done. The standard object wrapper class implements that.",
      {
	PyObject *callobj = unbox(prhs[0]);
	if (!PyCallable_Check(callobj))
	  mexErrMsgIdAndTxt("python:NotCallable", "tried to call object which is not callable.");
	PyObject *args = NULL;
	if (mxIsCell(prhs[1]))
	  args = mxCell_to_PyTuple(prhs[1]);
	else
	  args = unbox(prhs[1]);
	if (!args || !PyTuple_Check(args))
	  mexErrMsgIdAndTxt("python:NotTuple", "args must be a tuple");
	PyObject *kwargs = NULL;
	if (nrhs > 2) {
	  kwargs = unbox(prhs[2]);
	  if (kwargs && !PyDict_Check(kwargs))
	    mexErrMsgIdAndTxt("python:NoKWargs", "kwargs must be a dict or null");
	}
	#if PYMEX_DEBUG_FLAG
	PyObject *crepr = PyObject_Repr(callobj);
	PyObject *arepr = PyObject_Repr(args);
	PyObject *krepr = kwargs ? PyObject_Repr(kwargs) : NULL;
	mexPrintf("CALLOBJ: %s\n", PyBytes_AsString(crepr));
	mexPrintf("ARGS: %s\n", PyBytes_AsString(arepr));
	if (krepr)
	  mexPrintf("KEYS: %s\n", PyBytes_AsString(krepr));
	else
	  mexPrintf("KEYS: <null>\n");
	Py_XDECREF(crepr);
	Py_XDECREF(arepr);
	Py_XDECREF(krepr);
	#endif
	PyObject *result = PyObject_Call(callobj, args, kwargs);
	plhs[0] = box(result);
      })
#else
PYMEX(CALL, 2,3, 
      "Calls a callable python object. In addition to the "
      "object itself, the second argument is a cell array or tuple "
      "of arguments. An optional third argument is a dict of keyword arguments. "
      "No output unpacking is done. The standard object wrapper class implements that.",
      {
	PyObject *callobj = unbox(prhs[0]);
	if (!PyCallable_Check(callobj))
	  mexErrMsgIdAndTxt("python:NotCallable", "tried to call object which is not callable.");
	PyObject *args = NULL;
	if (mxIsCell(prhs[1]))
	  args = mxCell_to_PyTuple(prhs[1]);
	else
	  args = unbox(prhs[1]);
	if (!args || !PyTuple_Check(args))
	  mexErrMsgIdAndTxt("python:NotTuple", "args must be a tuple");
	PyObject *kwargs = NULL;
	if (nrhs > 2) {
	  kwargs = unbox(prhs[2]);
	  if (kwargs && !PyDict_Check(kwargs))
	    mexErrMsgIdAndTxt("python:NoKWargs", "kwargs must be a dict or null");
	}
	PyObject *result = PyObject_Call(callobj, args, kwargs);
	plhs[0] = box(result);
      })
#endif

PYMEX(IS_CALLABLE, 1,1, 
      "Tests the object to see if it is callable.",
      {
	plhs[0] = mxCreateLogicalScalar(PyCallable_Check(unbox(prhs[0])));
      })

PYMEX(GET_ATTR, 2,2, 
      "Gets the named attribute from the object.",
      {
	/* Perhaps we should use GetAttrString instead... */
	PyObject *pyobj = unbox(prhs[0]);
	PyObject *name = unboxn(prhs[1]);
	plhs[0] = box(PyObject_GetAttr(pyobj, name));
	Py_XDECREF(name);
      })

PYMEX(SET_ATTR, 3,3, 
      "Sets the named attribute. Argument order is name, value.",
      {    
	PyObject *pyobj = unbox(prhs[0]);
	PyObject *key = unboxn(prhs[1]);
	PyObject *val = unboxn(prhs[2]);
	PyObject_SetAttr(pyobj, key, val);
	Py_XDECREF(key);
	Py_XDECREF(val);
      })

PYMEX(HAS_ATTR, 2, 2, 
      "Asks the object whether it has a particular attribute.",
      {
	PyObject *pyobj = unbox(prhs[0]);
	PyObject *name = unboxn(prhs[1]);
	plhs[0] = mxCreateLogicalScalar(PyObject_HasAttr(pyobj, name));
	Py_XDECREF(name);
      })

PYMEX(GET_ITEM, 2,2, 
      "Retrieve an object using the given key.",
      {
	PyObject *pyobj = unbox(prhs[0]);
	PyObject *key = unboxn(prhs[1]);
	plhs[0] = box(PyObject_GetItem(pyobj, key));
	Py_XDECREF(key);
      })

PYMEX(SET_ITEM, 3,3, 
      "Sets an object using a given key. Argument order is key, value.",
      {
	PyObject *pyobj = unbox(prhs[0]);
	PyObject *key = unboxn(prhs[1]);
	PyObject *val = unboxn(prhs[2]);
	PyObject_SetItem(pyobj, key, val);
	Py_XDECREF(key);
	Py_XDECREF(val);
      })

PYMEX(GET_MODULE_DICT, 0,0, 
      "Retrieves the system's dictionary of imported modules.",
      {
	plhs[0] = boxb(PyImport_GetModuleDict());
      })

PYMEX(IS_INSTANCE, 2,2, 
      "For (obj, class), determines if object is an instance of class "
      "or one of its subclasses. class could also be a tuple of classes - "
      "see PyObject_IsInstance for details.",
      {
	plhs[0] = 
	  mxCreateLogicalScalar(PyObject_IsInstance(unbox(prhs[0]),
						    Any_mxArray_to_PyObject(prhs[1])));
      })

PYMEX(CELL_TO_TUPLE, 1,1, 
      "Just converts a cell array to a tuple.",
      {
	if (!mxIsCell(prhs[0]))
	  mexErrMsgIdAndTxt("pymex:NotCell", "This command only converts cells to tuples.");
	else
	  plhs[0] = box(mxCell_to_PyTuple(prhs[0]));
      })

PYMEX(TO_MXARRAY, 1,1,
      "Attempts to coerce a Python object to an appropriate MATLAB type.",
      {
	if (!mxIsPyObject(prhs[0]))
	  plhs[0] = (mxArray *) prhs[0];
	else
	  plhs[0] = Any_PyObject_to_mxArray(unbox(prhs[0]));
      })

PYMEX(VERSION, 0, 0,
      "Returns the git branch/tag where pymex was last built.",
      {
	plhs[0] = mxCreateString(CONST_TO_STR(PYMEX_BUILD));
      })
