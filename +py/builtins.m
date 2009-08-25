function b = builtins(varargin)
b = pymex(py.Interface.GET_BUILTINS);
if nargin > 0
    b = cellfun(@(name) getitem(b, name), varargin, 'uniformoutput', false);
    if nargin == 1
        b = b{1};
    else
        b = py.tuple(b{:});
    end
end

