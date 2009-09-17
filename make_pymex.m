function make_pymex(varargin)
try
    locked = pymex('MEXISLOCKED');
    if locked
        fprintf(2,'pymex is locked. Clear all PyObjects to unlock.\n');
        return
    end
catch %#ok<CTCH>
end
clear pymex;

dir = fileparts(which('make_pymex'));
if ~isequal(pwd, dir)
    fprintf('cd %s\n', dir);
    cd(dir);
end

if ~iscellstr(varargin)
    error('make takes string arguments only');
end

status = system(['make ' sprintf('%s ', varargin{:})]);
if status ~= 0
    fprintf(2, 'Failed with exit code %d\n', status);
else
    try
        c = pymex;
        fprintf('pymex loaded, and knows %d tricks.\n', numel(c));
    catch %#ok<CTCH>
        fprintf('pymex could not load.\n');
    end
end
