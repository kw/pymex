function v = float(val, varargin)
if numel(val) < 0
    val = 0;
end
if numel(val) == 1
    float = pybuiltins('float');
    v = float(val, varargin{:});
else
    v = py.array(val, 'float');
end
