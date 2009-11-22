from matlab import mx
from matlab import mex
import numbers
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

class _strfun(str):
    '''
    Use a string as a MATLAB function.
    The result is simply used as a function name to give
    to feval. To procure an actual function handle, use the
    function_handle class. 
    '''
    def __call__(self, *args, **kwargs):
        return mex.call(self, *args, **kwargs)


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

class _structel(object):
    '''
    Struct Element. Used by the struct class to allow
    MATLAB-like indexing like: foo(2,5).bar = 42
    The _structel returned by foo(2,5) redirects the
    __setattr__ call here to an appropriate call in the
    source struct. 

    _structel is also used for the default struct-like
    _object class, though this is only sufficient for
    the simplest classes and doesn't handle method calls.
    '''
    __slots__ = ('_source', '_ind')
    def __init__(self, source, ind):
        self._source = source
        self._ind = ind
    def __getattr__(self, key):
        if key in _structel.__slots__:
            return super(_structel, self).__getattr__(key)
        else:
            return self._source[self._ind, key]
    def __setattr__(self, key, val):
        if key in _structel.__slots__:
            super(_structel, self).__setattr__(key, val)
        else:
            self._source[self._ind, key] = val
    def __str__(self):
        return str(self._source[self._ind])
    def __repr__(self):
        return repr(self.__source[self._ind])

class struct(mx.Array):
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
            if len(self) == 1:
                return self._get_field(key, index=0)
            else:
                return [self._get_field(key, index=ind) for ind in range(len(self))]
        elif isinstance(key, numbers.Number):
            ind = _check_dims(self, int(key))
            return dict((f,self[ind,f]) for f in self._get_fields())
        else:
            raise KeyError, "I'm not sure what you want me to do with a key of type %s" % type(key)
    def __setitem__(self, key, val):
        if isinstance(key, tuple):
            (ind, key) = key
        elif isinstance(key, str) and len(self) == 1:
            ind = 0
        else:
            raise KeyError, "You'll have to set one index+field at a time: self[ind, field] = foo"
        ind = _check_dims(self, ind)
        self._set_field(key, val, index=ind)
    def __len__(self):
        return self._get_number_of_elements()
    def __call__(self, *ind):
        return _structel(self, _check_dims(self,ind))


class _object(mx.Array):
    '''
    The default representation for MATLAB class objects.
    Objects are assumed to be struct-like. This class does
    not attempt to call into MATLAB to do any work, instead
    relying on the only two C-level methods provided by MATLAB:
    mxGetProperty and mxSetProperty. With only these functions
    there is no way to verify that a property exists or what its
    access rights are, and querying a property that does not exist
    or that has the wrong rights can, in certain versions of MATLAB,
    cause immediate SIGABRT with no chance of recovery. Excellent design.
    
    Naturally we should provide a safer class, but it is not immediately
    obvious how that class should behave. Due to the bizarre nature of
    MATLAB's subsref/subsasgn facility it does not seem likely that we
    can provide an abstraction that works for all MATLAB classes and 
    doesn't leak all over the place. In case you're curious, subsref
    and subsasgn are to MATLAB as __get/setitem__, __get/setattr__,
    and __call__ are to Python, all at the same time. Not just in the
    sense that they are expected to handle all get/set functionality,
    but also in the sense that any compound get or set is handled as
    a single call. So if you had something like, say:
    x = foo{1,2}.bar{3}.spam(4,:);, a single subsref would be called on
    the first object (foo), and it would be given a struct array with entries:
    S(1).type = '{}', .subs = {1, 2}
    S(2).type = '.', .subs = {'bar'}
    S(3).type = '{}', .subs = {3}
    S(4).type = '.', .subs = {'spam'}
    S(5).type = '()', .subs = {4, ':'}  (and yes, that is the colon *character*)
    The object foo is thus required to handle the entire expression. Normally one
    would implement this sanely by doing the first item and then passing the rest
    of the expression on to the result, but it is perfectly permissible for foo
    to fake the whole thing. subsasgn works similarly. This is insane, and at first
    glance I'd say that this is not implementable in a sufficiently general way in
    Python. It doesn't help that they have two different sets of indexing brackets,
    one of which is *also* used to indicate function calls *and* is a valid target
    for assignment. I won't even touch the colon, end, or subsindex issues.

    I'll probably implement something non-transparent but still
    usable that works for most classes. But this is what we have for now,
    and I'd recommend that if you need a particular class to work a particular way,
    that you use the mltypes package to implement it and let me know how it goes.
    '''
    def __getitem__(self, key):
        if isinstance(key, tuple):
            (ind, prop) = key
            ind = _check_dims(self, ind)
            return self._get_property(prop, index=ind)
        elif isinstance(key, str):
            if len(self) == 1:
                return self._get_property(key, index=0)
            else:
                return [self._get_property(key, index=ind) for ind in range(len(self))]
        ## We have no _get_properties function because MATLAB doesn't provide it.
        ## Could probably try calling the "properties" method, but that's a mex call.
        #elif isinstance(key, numbers.Number):
        #    ind = _check_dims(self, int(key))
        #    return dict((p,self[ind,p]) for p in self._get_properties())
        else:
            raise KeyError, "I'm not sure what you want me to do with a key of type %s" % type(key)
    def __setitem__(self, key, val):
        if isinstance(key, tuple):
            (ind, key) = key
        elif isinstance(key, str) and len(self) == 1:
            ind = 0
        else:
            raise KeyError, "You'll have to set one index+prop at a time: self[ind, field] = foo"
        ind = _check_dims(self, ind)
        self._set_property(key, val, index=ind)
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, val):
        self[key] = val
    def __len__(self):
        return self._get_number_of_elements()
    def __call__(self, *ind):
        return _structel(self, _check_dims(self,ind))

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

