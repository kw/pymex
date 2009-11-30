'''
This module is an example wrapper for MATLAB's 'containers' package,
which at last count only contained the Map class, which behaves sort
of oddly but more or less works. Likewise, a Map class is provided here,
and any containers.Map object that makes its way into Python will be
wrapped in mltypes.containers.Map. 
'''
# Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
# For full license details, see the LICENSE file.
import mx
from matlab import substruct, subsref, subsasgn, isKey, keys, values, remove

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
    def __contains__(self, key):
        return bool(isKey(self, key))
    def keys(self):
        return keys(self)
    def values(self):
        return values(self)

        
