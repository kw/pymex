#include <Python.h>
#include "matrix.h"
#include "structmember.h"

typedef struct {
    PyObject_HEAD
    mxArray* mxobj;
} Py_mxArray;

static void 
Py_mxArray_dealloc(Py_mxArray* self) {
  if (self->mxobj)
    mxDestroyArray(self->mxobj);
  self->ob_type->tp_free((PyObject*) self);
}

static PyObject* 
Py_mxArray_new(PyTypeObject *type, PyObject* args, PyObject* kwds) {
  Py_mxArray* self;
  self = (Py_mxArray*) type->tp_alloc(type, 0);
  if (self != NULL) {
    self->mxobj = NULL;
  }
  return (PyObject*) self;
}

static int
Py_mxArray_init(Py_mxArray* self, PyObject* args, PyObject* kwds) {
  PyObject* source = NULL;
  PyObject* size = NULL;
  mxClassID class = mxDOUBLE_CLASS;
  mxComplexity complex = mxREAL;
  static char *kwlist[] = {"size", "class", "complex", "source"};
  if (! PyArg_ParseTupleAndKeywords(args, kwds, 
				    "|NiiN", kwlist,
				    &size, &class, &complex, &source))
    return -1;
  if (size && source) {
    return -1;
  }
  else if (size) {
    mwSize len = PySequence_Length(size);
    mwSize dims[len];
    mwSize i;
    for (i=0; i<len; i++) {
      PyObject* item = PySequence_GetItem(size, i);
      dims[i] = (mwSize) PyNumber_AsSsize_t(item, NULL);
      Py_DECREF(item);
    }
    self->mxobj = mxCreateNumericArray(len, dims, class, complex);
  }
  else if (source) {
    if (!PyCObject_Check(source)) return -1;
    self->mxobj = (mxArray*) PyCObject_AsVoidPtr(source);
  }
  else
    return -1;
  return 0;
}

static PyObject*
Py_mxArray_mxclass(Py_mxArray* self) {
  const char* classname = mxGetClassName(self->mxobj);
  return PyString_FromString(classname);
}

static PyObject*
Py_mxArray_mxsize(Py_mxArray* self) {
  const mwSize nd = mxGetNumberOfDimensions(self->mxobj);
  const mwSize* dims = mxGetDimensions(self->mxobj);
  PyObject* tup = PyTuple_New(nd);
  if (!tup) return NULL;
  long i;
  for (i=0; i<nd; i++) {
    PyObject* d = PyLong_FromLong((long) dims[i]);
    PyTuple_SetItem(tup, i, d);
  }
  return tup;
}

static PyMethodDef Py_mxArray_methods[] = {
  {"mxclass", (PyCFunction) Py_mxArray_mxclass, METH_NOARGS,
   "Returns the name of the mxArray's internal class."},
  {"mxsize", (PyCFunction) Py_mxArray_mxsize, METH_NOARGS,
   "Returns the size of the mxArray along each dimension."},
    {NULL}  /* Sentinel */
};

static PyTypeObject Py_mxArrayType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /*ob_size*/
  "libmx.mxArray",           /*tp_name*/
  sizeof(Py_mxArray),        /*tp_basicsize*/
  0,                         /*tp_itemsize*/
  (destructor)Py_mxArray_dealloc,   /*tp_dealloc*/
  0,                         /*tp_print*/
  0,                         /*tp_getattr*/
  0,                         /*tp_setattr*/
  0,                         /*tp_compare*/
  0,                         /*tp_repr*/
  0,                         /*tp_as_number*/
  0,                         /*tp_as_sequence*/
  0,                         /*tp_as_mapping*/
  0,                         /*tp_hash */
  0,                         /*tp_call*/
  0,                         /*tp_str*/
  0,                         /*tp_getattro*/
  0,                         /*tp_setattro*/
  0,                         /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /*tp_flags*/
  "mxArray objects",         /* tp_doc */
  0,		             /* tp_traverse */
  0,		             /* tp_clear */
  0,		             /* tp_richcompare */
  0,		             /* tp_weaklistoffset */
  0,		             /* tp_iter */
  0,		             /* tp_iternext */
  Py_mxArray_methods,        /* tp_methods */
  0,                         /* tp_members */
  0,                         /* tp_getset */
  0,                         /* tp_base */
  0,                         /* tp_dict */
  0,                         /* tp_descr_get */
  0,                         /* tp_descr_set */
  0,                         /* tp_dictoffset */
  (initproc)Py_mxArray_init, /* tp_init */
  0,                         /* tp_alloc */
  Py_mxArray_new,            /* tp_new */
};

static PyMethodDef libmx_methods[] = {
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initlibmx(void) 
{
    PyObject* m;

    if (PyType_Ready(&Py_mxArrayType) < 0)
        return;

    m = Py_InitModule3("libmx", libmx_methods,
                       "libmx, the MATLAB matrix type library.");

    Py_INCREF(&Py_mxArrayType);
    PyModule_AddObject(m, "mxArray", (PyObject *)&Py_mxArrayType);
}
