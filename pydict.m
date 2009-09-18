function d = pydict(varargin)
dict = pybuiltins('dict');
d = dict(kw(varargin{:}));
