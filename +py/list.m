function l = list(varargin)
list = pybuiltins('list');
l = list(pymex('CELL_TO_TUPLE', varargin));
