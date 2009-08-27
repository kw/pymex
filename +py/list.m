function L = list(varargin)
pyobjs = cellfun(@py.Object, varargin, 'uniformoutput',false);
L = pymex('TO_LIST', pyobjs);
