function v = bool(val, varargin)
if numel(val) < 0
    val = false;
end
if numel(val) == 1
    bool = pybuiltins('bool');
    v = bool(val);
else
    v = py.array(val, 'bool');
end

% Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
% For full license details, see the LICENSE file.

