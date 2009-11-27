import matlab.mx as mx

try: import numpy as np
except: pass

_unpy_registry = dict()

def register_unpy(key, func):
    _unpy_registry[key] = func

def unregister_unpy(key):
    del _unpy_registry[key]

def unpy(obj):
    # Check if object knows how to do it
    try: return obj.__mxArray__()
    except: pass
    # Check if object is registered
    try: return _unpy_registry[obj](obj)
    except: pass
    # Check if class is registered
    try:
        for cls in type(obj).mro():
            try: return _unpy_registry[cls](obj)
            except: pass
    except: pass
    raise NotImplementedError

_dtype_map = None
def select_mxclass_by_dtype(dtype):
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
    mxclass = select_mxclass_by_dtype(self.dtype.type)
    cobj = mx.create_numeric_array(mxclass = mxclass,
                                   dims = self.shape)
    wrapper = mx.wrap_pycobject(cobj)
    wrap_array = np.asarray(wrapper)
    wrap_array[...] = self
    return wrapper

# The try block here is in case we don't actually have numpy.
try: register_unpy(np.ndarray, numpy_ndarray_unpy)
except: pass
