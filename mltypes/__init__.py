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


class cell(mx.Array):
    def _check_dims(self, ind):
        if isinstance(ind, tuple):
            if any(map(lambda a: isinstance(a, slice), ind)):
                raise KeyError, "slicing not yet supported"
            numinds = len(ind)
            dims = self.get_dimensions()
            if numinds > len(dims): 
                raise KeyError, "too many dimensions"
            elif numinds < len(dims):
                shortdims = dims[:numinds]
                shortdims.append(reduce(lambda a,b: a*b, dims[numinds:]))
                dims = shortdims;
            if any(map(lambda a,b: b <= long(a), ind, dims)):
                raise KeyError, "at least one index out of bounds"
            # all seems well, so calc the real index
            ind = self.calc_single_subscript(*ind)
        elif isinstance(ind, slice):
            raise KeyError, "slicing not yet supported"
        if ind > len(self):
            raise KeyError, "linear index out of bounds"
        return ind
    def __getitem__(self, ind):
        ind = self._check_dims(ind)
        return self.get_cell(ind)
    def __setitem__(self, ind, val):
        ind = self._check_dims(ind)
        self.set_cell(ind, val)
    def __len__(self):
        return self.get_number_of_elements()


class struct(mx.Array):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, val):
        self[key] = val
    def __getitem__(self, key):
        ind = 0
        if isinstance(key, tuple):
            (ind, key) = key
        return self.get_field(key, index=ind)
    def __setitem__(self, key, val):
        ind = 0
        if isinstance(key, tuple):
            (ind, key) = key
        self.set_field(key, val, index=ind)
    def __len__(self):
        return self.get_number_of_elements()
