function v = pycall(builtin, varargin)
fun = pybuiltins(builtin);
v = fun(varargin{:});
