'''
Pymex Utility module
This module is primarily for internal use, though users
who need to modify some behaviors (particularly with type conversion)
might want to have a look-see. 
'''

import mx
import struct

__unpy_registry = dict()
def register_unpy(cls, func):
    '''
    Provide a function to convert members of the
    given class to (or towards) mxArray. See 'unpy'.
    '''
    __unpy_registry[cls] = func

def unregister_unpy(cls):
    '''
    Remove a registered converter function.
    '''
    del __unpy_registry[cls]
    
def unpy(obj):
    '''
    Converts the given object to an mxArray 
    This is called by Any_PyObject_to_mxArray in the C sources.

    To make your types work with this, provide a
    __mxArray__() method. You may be able to inject
    this into classes that you don't control, but otherwise
    you can use register_unpy(cls, func) to register it.
    If nothing appropriate can be found, raises NotImplementedError
    '''
    # Check if object knows how to do it
    try: return obj.__mxArray__()
    except: pass
    # Check if class is registered
    for cls in type(obj).mro():
            if cls in __unpy_registry:
                return __unpy_registry[cls](obj)
    raise NotImplementedError

## unpy handlers

def str_unpy(self):
    '''
    Converts str to mxCHAR_CLASS via libmx
    '''
    return mx.create_string(self, wrap=True)
register_unpy(str, str_unpy)


def seq_to_cell(self):
    '''
    Given an iterable of known length, produce a cell array.
    This is not done recursively. We would need to check for
    recursive structure in that case.
    '''
    size = (1, len(self))
    cell = mx.create_cell_array(size, wrap=True)
    i = 0
    for item in self:
        cell[i] = item
        i += 1
    return cell

register_unpy(tuple, seq_to_cell)
register_unpy(list, seq_to_cell)

def _make_scalar_unpy(scalartype, fmt, mxclass):
    def _scalar_unpy(self):
        byteval = struct.pack(fmt, self)
        scalar = mx.create_numeric_array(dims=(1,1), mxclass=mxclass, wrap=True)
        assert len(byteval) == scalar._get_element_size(), "Datatype size mismatch"
        scalar._set_element(byteval)
        return scalar
    register_unpy(scalartype, _scalar_unpy)

try: 
    _make_scalar_unpy(long, "q", mx.INT64)
    _make_scalar_unpy(int, "i", mx.INT32)
except: # Python 3 does not have longs. ints are long when necessary.
    _make_scalar_unpy(int, "q", mx.INT64)

_make_scalar_unpy(float, "d", mx.DOUBLE)
_make_scalar_unpy(bool, "?", mx.LOGICAL)


## Some sample unpy stuff for numpy 
try: 
    import numpy as np

    __dtype_map = None
    def select_mxclass_by_dtype(dtype):
        '''
        Tries to select an appropriate mxclass.
        Doesn't handle complex dtypes.
        '''
        global __dtype_map
        if __dtype_map is None:
            __dtype_map = {
                np.float64 : mx.DOUBLE,
                np.float32 : mx.SINGLE,
                np.int64 : mx.INT64,
                np.uint64 : mx.UINT64,
                np.int32 : mx.INT32,
                np.uint32 : mx.UINT32,
                np.int16 : mx.INT16,
                np.uint16 : mx.UINT16,
                np.int8 : mx.INT8,
                np.uint8 : mx.UINT8,
                np.bool8 : mx.LOGICAL,
                np.character : mx.CHAR,
                }
        # First handle the common cases
        try: return __dtype_map[dtype]
        except: pass
        # Make a vague attempt at providing something appropriate
        if issubclass(dtype, np.floating): return mx.DOUBLE
        elif issubclass(dtype, np.unsignedinteger): return mx.UINT64
        elif issubclass(dtype, np.integer): return mx.INT64
        else: raise TypeError, "Couldn't figure out an appropriate mxclass for dtype %r" % dtype

    def numpy_ndarray_unpy(self):
        '''
        Converts a numpy array to an mxArray. An appropriate
        data type is selected using select_mxclass_by_dtype.
        Fattens the array or scalar out to 2d if necessary. 
        FIXME: Doesn't do anything appropriate for complex
        dtypes.
        '''
        self = np.atleast_2d(self)
        mxclass = select_mxclass_by_dtype(self.dtype.type)    
        cobj = mx.create_numeric_array(mxclass = mxclass,
                                       dims = self.shape)
        wrapper = mx.wrap_pycobject(cobj)
        wrap_array = np.asarray(wrapper)
        wrap_array[...] = self
        return wrapper

    register_unpy(np.ndarray, numpy_ndarray_unpy)
    # Also register it for numpy scalars
    register_unpy(np.generic, numpy_ndarray_unpy)
except: pass # No numpy for you, I guess


