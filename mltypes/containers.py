import matlab.mx as mx
import mltypes

substruct = mltypes._strfun("substruct")
subsref = mltypes._strfun("subsref")
subsasgn = mltypes._strfun("subsasgn")
isKey = mltypes._strfun("isKey")
keys = mltypes._strfun("keys")
values = mltypes._strfun("values")
remove = mltypes._strfun("remove")

class Map(mx.Array):    
    def __getitem__(self, key):
        if not isinstance(key, tuple): key = (key,)
        struct = substruct('()', key)
        return subsref(self, struct)
    def __setitem__(self, key, val):
        if not isinstance(key, tuple): key = (key,)
        struct = substruct('()', key)
        subsasgn(self, struct, val)
    def __delitem__(self, key):
        remove(self, key)
    def __len__(self):
        return int(self._get_property('Count'))
    def __contains__(self, key): # FIXME: the 'in' operator always returns True?
        return bool(isKey(self, key))
    def keys(self):
        return keys(self)
    def values(self):
        return values(self)

        
