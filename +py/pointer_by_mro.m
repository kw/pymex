% Get most appropriate pointer object based on MRO list
% This is called directly from pymex. Users should probably ignore it.
function pyobj = pointer_by_mro(mro)
for i=1:numel(mro)
    try
        pyobj = feval(['py.types.' mro{i}]);
        return;
    catch %#ok        
    end
end
pyobj = py.types.object();
