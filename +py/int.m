function n = int(varargin)
int = pybuiltins('int');
if nargin > 0
    n = int(varargin{:});
else
    n = int;
end
