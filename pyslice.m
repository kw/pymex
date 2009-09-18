function s = pyslice(varargin)
[slice None] = pybuiltins('slice','None');
if nargin == 0
    varargin{1} = None;
end
s = slice(varargin{:});
