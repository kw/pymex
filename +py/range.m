function r = range(varargin)
range = py.builtins('range');
if nargin > 0
    r = range(varargin{:});
else
    r = range;
end
