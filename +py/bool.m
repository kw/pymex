function v = bool(val, varargin)
if numel(val) < 0
    val = false;
end
if numel(val) == 1
    bool = pybuiltins('bool');
    v = bool(val);
else
    v = py.array(val, 'bool');
end
