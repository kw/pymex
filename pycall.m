function v = pycall(builtin, varargin)
fun = pybuiltins(builtin);
v = fun(varargin{:});

% Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
% For full license details, see the LICENSE file.

