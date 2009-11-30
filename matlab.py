'''
This module is evil.
It redefines __getattr__ in such a way that things
not actually in the module's dict are turned into
MATLAB function handles. These are cached for future
use. You could use it like:
    import matlab
    matlab.fprintf("spam\n") # Possibly better than using mex.printf

Or as in mltypes.containers:
from matlab import substruct, subsref, subsasgn, isKey, keys, values, remove

This module seems to work for the moment, but it lacks in two
areas: you can't do `from matlab import *`, (or at least, if you do then
you only get previously referenced items), and you can't reference things
in MATLAB packages. I'll try to replace this at a later time with some
import hooks. They will also be evil, but in other ways.
'''
# Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
# For full license details, see the LICENSE file.

import sys as _sys
from mltypes import function_handle as _function_handle

class __MATLABModule(_sys.__class__):
    def __init__(self, other):
        for attr in dir(other):
            val = getattr(other, attr)
            setattr(self, attr, val)
            assert getattr(self, attr) is val
    def __getattr__(self, name):
        if name in dir(self):
            return getattr(self, name)
        elif name.startswith('_'):
            raise AttributeError, "MATLAB names can't start with underscores."
        else:
            # While function_handle is a more heavy-weight
            # operation than mltypes._strfun, it is theoretically
            # faster to call function handles, so it is probably
            # acceptable to have higher up-front cost. Especially
            # since we're caching the result in this module.
            strfun = self._function_handle(name=name)
            setattr(self, name, strfun)
            return strfun


_sys.modules[__name__] = __MATLABModule(_sys.modules[__name__])
