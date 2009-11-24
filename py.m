function varargout = py(varargin)
varargout = cell(size(varargin));
for i=1:numel(varargin)
    varargout{i} = pymex('TO_PYOBJECT', varargin{i});
end

% Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
% For full license details, see the LICENSE file.

