function L = list(varargin)
pyobjs = cellfun(@py.Object, varargin, 'uniformoutput',false);
L = pymex(py.Interface.TO_LIST, pyobjs);
