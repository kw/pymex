function d = dir(pyobj)
if nargin < 1
    pyobj = py.Object();
end
d = pymex(py.Interface.DIR, pyobj);