#ifndef PYMEX_HELPERS_INCLUDED
#define PYMEX_HELPERS_INCLUDED
#include <Python.h>
#if PYMEX_USE_NUMPY
 #ifndef PY_ARRAY_UNIQUE_SYMBOL
  #define PY_ARRAY_UNIQUE_SYMBOL PYMEX_ARRAY_API
  #define NO_IMPORT_ARRAY
  #define NPY_USE_PYMEM 1
  #include <numpy/arrayobject.h>
 #endif
#endif 

#define PY3K_VERSION_HEX 0x3000000

#define PYMEX_MATLAB_VOIDPTR "py.types.voidptr"
#define PYMEX_MATLAB_PYOBJECT "py.types.builtin.object"

/* MATLAB's matrix type library. 
   Don't necessarily want to include the full mex.h everywhere,
   but this should provide all the necessary API for MATLAB types.
*/
#include <matrix.h>

#ifndef PYMEX_DEBUG_FLAG
#define PYMEX_DEBUG_FLAG 0
#endif

#if PYMEX_DEBUG_FLAG
#define PYMEX_DEBUG(format, args...) mexPrintf(format,##args)
#else
#define PYMEX_DEBUG(format, args...) /*nop*/
#endif

/* PYMEX_SQUEEZE_SMALL_ARRAYS
   MATLAB does not allow arrays to have fewer than 2 dimensions.
   Numpy, on the other hand, allows 1-d or even 0-d arrays (scalars).
   When converting from Numpy to MATLAB arrays, we must naturally increase
   the number of dimensions for small-dimensioned arrays, but going the other
   direction isn't necessarily clear.
   If PYMEX_SQUEEZE_SMALL_ARRAYS is true, pymex will detect singleton dimensions
   of a 2D array and remove them. This might make broadcasting easier, but
   it also causes vectors to forget their orientation. 
 */
#ifndef PYMEX_SQUEEZE_SMALL_ARRAYS
#define PYMEX_SQUEEZE_SMALL_ARRAYS 0
#endif

/* PYMEX_SCALARIZE_SCALARS
   Similarily, we can extract numpy scalars from 0-d arrays when appropriate. 
   Unfortunately, numpy scalars don't quite have the same semantics as ndarrays,
   so certain operations (like concatenation) don't work right. Note this only affects
   0-d arrays, not arrays with 1 element, so if PYMEX_SQUEEZE_SMALL_ARRAYS is false
   then this won't affect arrays obtained from MATLAB (since they'll be at least 2-d)
 */
#ifndef PYMEX_SCALARIZE_SCALARS
#define PYMEX_SCALARIZE_SCALARS 0
#endif

/* Use this in place of PyArray_Return */
#if PYMEX_SCALARIZE_SCALARS
#define PYMEX_PYARRAY_RETURN(pyobj) PyArray_Return((PyArrayObject*) pyobj)
#else
#define PYMEX_PYARRAY_RETURN(pyobj) pyobj
#endif

mxArray* box(PyObject* pyobj);
mxArray* boxb(PyObject* pyobj);
PyObject* unbox (const mxArray* mxobj);
PyObject* unboxn (const mxArray* mxobj);
bool mxIsPyNull (const mxArray* mxobj);
bool mxIsPyObject(const mxArray* mxobj);
mxArray* PyObject_to_mxLogical(PyObject* pyobj);
PyObject* mxChar_to_PyString(const mxArray* mxchar);
PyObject* mxCell_to_PyTuple(const mxArray* mxobj);
mxArray* PyString_to_mxChar(PyObject* pystr);
mxArray* PyObject_to_mxChar(PyObject* pyobj);
mxArray* PySequence_to_mxCell(PyObject* pyobj);
mxArray* PyObject_to_mxDouble(PyObject* pyobj);
mxArray* PyObject_to_mxLong(PyObject* pyobj);
PyObject* Any_mxArray_to_PyObject(const mxArray* mxobj);
mxArray* Any_PyObject_to_mxArray(PyObject* pyobj);
bool PyMXObj_Check(PyObject* pyobj);
PyObject* Py_mxArray_New(const mxArray* mxobj, bool duplicate);
int Py_mxArray_Check(PyObject* pyobj);
PyObject* mxArray_to_PyArray(const mxArray* mxobj, bool duplicate);
mxArray* PyArray_to_mxArray(PyObject* pyobj);
PyMODINIT_FUNC initmatlabmodule(void);
PyMODINIT_FUNC initmexmodule(void);
PyMODINIT_FUNC initmxmodule(void);
PyMODINIT_FUNC initmatmodule(void);
PyMODINIT_FUNC initengmodule(void);
char mxClassID_to_Numpy_Typekind(mxClassID mxclass);


#ifndef MEXMODULE
extern PyObject* mexmodule;
#else
PyObject* mexmodule;
#endif

#ifndef MXMODULE
extern PyObject* mxmodule;
#else
PyObject* mxmodule;
#endif

#ifndef MATLABMODULE
extern PyObject* matlabmodule;
#else
PyObject* matlabmodule;
#endif

#ifndef MATMODULE
extern PyObject* matmodule;
#else
PyObject* matmodule;
#endif

#ifndef ENGMODULE
extern PyObject* engmodule;
#else
PyObject* engmodule;
#endif

#define mxArrayPtr(pyobj) ((mxArray*) ((mxArrayObject*)pyobj)->mxptr)

typedef struct {
    PyObject_HEAD
    mxArray* mxptr;
} mxArrayObject;

#endif
