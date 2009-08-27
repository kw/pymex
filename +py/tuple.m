function L = tuple(varargin)
pyobjs = cellfun(@py.Object, varargin, 'uniformoutput',false);
L = pymex('TO_TUPLE', pyobjs);
