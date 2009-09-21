function v = int(val, varargin)
if numel(val) < 0
    val = 0;
end
if numel(val) == 1
    int = pybuiltins('int');
    v = int(val, varargin{:});
else
    v = py.array(val, 'int');
end

