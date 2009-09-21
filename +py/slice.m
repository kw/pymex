function s = slice(varargin)
[slice None] = pybuiltins('slice','None');
if nargin == 0
    varargin{1} = None;
end
for i=1:numel(varargin)
    if isempty(varargin{i})
        varargin{i} = None;
    end
end
s = slice(varargin{:});
