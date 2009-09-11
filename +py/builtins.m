function varargout = builtins(varargin)
b = pymex('GET_BUILTINS');
if nargin > 0
    b = cellfun(@(name) getitem(b, name), varargin, 'uniformoutput', false);
    if nargin == 1 && nargout == 1
        varargout{1} = b{1};
    elseif nargin > 1 && nargout == 1
        varargout{1} = py.tuple(b);
    elseif nargin > 1 && nargout > 1
        varargout{1:nargout} = b{1:nargout};
    elseif nargout == 0
        for i=1:numel(varargin)
            assignin('caller', varargin{i}, b{i});
        end                
    end
elseif nargout == 0
    keys = b.keys();
    values = b.values();
    inds = 0:len(keys);
    for i=inds
        name = char(keys{i});
        if isvarname(name)
            assignin('caller', name, values{i});
        end
    end
else
    varargout{1} = b;
end
