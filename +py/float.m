function n = float(varargin)
float = pybuiltins('float');
if nargin > 0
    n = float(varargin{:});
else
    n = float;
end
