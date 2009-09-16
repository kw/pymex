function s = SI(varargin)
slice = pybuiltins('slice');
for s=1:numel(varargin)
    if isfloat(varargin{s})
        varargin{s} = int64(varargin{s});
    end
end
s = slice(varargin{:});
