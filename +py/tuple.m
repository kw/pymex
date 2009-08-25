function L = tuple(varargin)
pyobjs = cellfun(@py.Object, varargin, 'uniformoutput',false);
L = pymex(py.Interface.TO_TUPLE, pyobjs);
