import matlab.mx as mx

_unpy_registry = dict()

def register_unpy(cls, func):
    '''
    Provide a function to convert members of the
    given class to (or towards) mxArray. See 'unpy'.
    '''
    _unpy_registry[cls] = func

def unregister_unpy(cls):
    '''
    Remove a registered converter function.
    '''
    del _unpy_registry[cls]
    
def unpy(obj):
    '''
    Converts the given object to an mxArray, or perhaps
    one step closer than it was. This is called by
    Any_PyObject_to_mxArray in the C sources.

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
            if cls in _unpy_registry:
                return _unpy_registry[cls](obj)
    raise NotImplementedError

## Some sample unpy stuff for numpy 
try: 
    import numpy as np

    _dtype_map = None
    def select_mxclass_by_dtype(dtype):
        '''
        Tries to select an appropriate mxclass.
        Doesn't handle complex dtypes.
        '''
        global _dtype_map
        if _dtype_map is None:
            _dtype_map = {
                np.float64 : mx.mxDOUBLE_CLASS,
                np.float32 : mx.mxSINGLE_CLASS,
                np.int64 : mx.mxINT64_CLASS,
                np.uint64 : mx.mxUINT64_CLASS,
                np.int32 : mx.mxINT32_CLASS,
                np.uint32 : mx.mxUINT32_CLASS,
                np.int16 : mx.mxINT16_CLASS,
                np.uint16 : mx.mxUINT16_CLASS,
                np.int8 : mx.mxINT8_CLASS,
                np.uint8 : mx.mxUINT8_CLASS,
                np.bool8 : mx.mxLOGICAL_CLASS,
                np.character : mx.mxCHAR_CLASS,
                }
        # First handle the common cases
        try: return _dtype_map[dtype]
        except: pass
        # Make a vague attempt at providing something appropriate
        if issubclass(dtype, np.floating): return mx.mxDOUBLE_CLASS
        elif issubclass(dtype, np.unsignedinteger): return mx.mxUINT64_CLASS
        elif issubclass(dtype, np.integer): return mx.mxINT64_CLASS
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


