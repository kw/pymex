function n = float(varargin)
float = py.builtins('float');
if nargin > 0
    n = float(varargin{:});
else
    n = float;
end
