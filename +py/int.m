function n = int(varargin)
int = py.builtins('int');
if nargin > 0
    n = int(varargin{:});
else
    n = int;
end
