function make(varargin)
try
    locked = pymex('MEXISLOCKED');
    if locked
        fprintf(2,'pymex is locked. Clear all py.Objects to unlock.\n');
        return
    end
catch %#ok<CTCH>
end
clear pymex;

if ~iscellstr(varargin)
    error('make takes string arguments only');
end

system(['make ' sprintf('%s ', varargin{:})]);

try
    c = pymex;
    fprintf('pymex built with %d system calls.\n', numel(c));
catch %#ok<CTCH>
    fprintf(2,'pymex build failed somehow\n');
end
