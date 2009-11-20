from matlab import mx
from matlab import mex
import sys

def _findtype(typelist):
    for (modstr, clsstr) in typelist:
        if len(modstr): 
            adjusted_modstr = "mltypes.%s" % modstr
        else:
            adjusted_modstr = "mltypes"
            try:
                tempmod = __import__(adjusted_modstr, globals(), locals(), [clsstr], 0)
                cls = getattr(tempmod, clsstr);
                return cls;
            except:
                pass            
    # if we got this far and found nothing, just use mx.Array
    return mx.Array;

def _check_dims(self, ind):
    if isinstance(ind, tuple) and len(ind) > 1:
        if any(map(lambda a: isinstance(a, slice), ind)):
            raise KeyError, "slicing not yet supported"
        numinds = len(ind)
        dims = self._get_dimensions()
        if numinds > len(dims): 
            raise KeyError, "too many dimensions"
        elif numinds < len(dims):
            shortdims = list(dims[:numinds])
            shortdims.append(reduce(lambda a,b: a*b, dims[numinds:]))
            dims = shortdims;
        if any(map(lambda a,b: b <= a, ind, dims)):
            raise KeyError, "at least one index out of bounds"
        # all seems well, so calc the real index
        ind = self._calc_single_subscript(*ind)
    elif isinstance(ind, tuple) and len(ind) == 1:
        ind = ind[0]
    elif isinstance(ind, slice):
        raise KeyError, "slicing not yet supported"
    ind = int(ind) # mxArray doesn't do comparisons yet...
    if ind > len(self):
        raise KeyError, "linear index out of bounds: %d > %d" % (ind, len(self))
    return ind

class cell(mx.Array):
    def __init__(self, dims=(1,1), mxpointer=None):
        if mxpointer is None:
            mxpointer = mx.create_cell_array(dims)
        super(cell, self).__init__(mxpointer=mxpointer)
    def __getitem__(self, ind):
        ind = _check_dims(self, ind)
        return self._get_cell(ind)
    def __setitem__(self, ind, val):
        ind = _check_dims(self, ind)
        self._set_cell(ind, val)
    def __len__(self):
        return self._get_number_of_elements()

class struct(mx.Array):
    class _structel(object):
        __slots__ = ('_source', '_ind')
        def __init__(self, source, ind):
            self._source = source
            self._ind = ind
        def __getattr__(self, key):
            if key in struct._structel.__slots__:
                return super(struct._structel, self).__getattr__(key)
            else:
                return self._source[self._ind, key]
        def __setattr__(self, key, val):
            if key in struct._structel.__slots__:
                super(struct._structel, self).__setattr__(key, val)
            else:
                self._source[self._ind, key] = val
        def __str__(self):
            return str(self._source[self._ind])
        def __repr__(self):
            return repr(self.__source[self._ind])
    def __init__(self, dims=(1,1), mxpointer=None):
        if mxpointer is None:
            mxpointer = mx.create_struct_array(dims)
        super(struct, self).__init__(mxpointer=mxpointer)
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, val):
        self[key] = val
    def __getitem__(self, key):
        if isinstance(key, tuple):
            (ind, field) = key
            ind = _check_dims(self, ind)
            return self._get_field(field, index=ind)
        elif isinstance(key, str):
            return [self._get_field(key, index=ind) for ind in range(len(self))]
        elif not(isinstance(key, str)):
            ind = _check_dims(self, int(key))
            return dict((f,self[ind,f]) for f in self._get_fields())
        else:
            raise KeyError, "I'm not sure what you want me to do with a key of type %s" % type(key)
    def __setitem__(self, key, val):
        if isinstance(key, tuple):
            (ind, key) = key
        else:
            raise KeyError, "You'll have to set one index+field at a time: self[ind, field] = foo"
        ind = _check_dims(self, ind)
        self._set_field(key, val, index=ind)
    def __len__(self):
        return self._get_number_of_elements()
    def __call__(self, *ind):
        return struct._structel(self, _check_dims(self,ind))


class _numeric(mx.Array):
    def __cmp__(self, other):
        # probably good enough for now...
        floatval = float(self)
        return cmp(floatval, other)


class function_handle(mx.Array):
    def __init__(self, name=None, closure=None, mxpointer=None):
        if mxpointer is None:
            if name and closure:
                raise ValueError, "Specify name OR closure"
            elif name:
                mxpointer = mx.create_function_handle(name=name)
            elif closure:
                mxpointer = mx.create_function_handle(closure=closure)
            else:
                raise ValueError, "Must specify a function name or MATLAB closure literal"
        super(function_handle, self).__init__(mxpointer=mxpointer)
    def __call__(self, *args, **kwargs):
        return mex.call(self, *args, **kwargs)

