function t = tuple(varargin)
tuple = pybuiltins('tuple');
t = tuple(pymex('CELL_TO_TUPLE', varargin));
