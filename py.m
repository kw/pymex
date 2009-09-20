function varargout = py(varargin)
varargout = cell(size(varargin));
for i=1:numel(varargin)
    varargout{i} = pymex('TO_PYOBJECT', varargin{i});
end
