% mrolist = mro(object, addvirtual=true, autosplit=true)
% Determines something similar to the method resolution order for an object.
% Note that this is probably not the actual order, since a cursory inspection
% did not reveal any clues as to what the correct order might be. So this just
% produces a list of all of the object's classes from most to least specific.
%
% If addvirtual is true, a virtual class may be appended to the end of the
% mro list to indicate the nonexistent root of a particular hierarchy. 
% The virtual classes are:
%  _numeric:   Numbers, logicals, and characters.
%  _object:    MATLAB objects
%  _java:      Java objects
%  _com:       COM objects (windows only)
%  _interface: COM interface (windows only)
% 
% Note that no attempt is made to enumerate the class hierarchy
% of Java classes. We could probably do that, but let's keep it simple for now.
function [mrolist] = mro(object,addvirtual,autosplit)
if nargin < 2, addvirtual = true; end
if nargin < 3, autosplit = true; end
classname = class(object);
mrolist = {classname};
classqueue = {classname};
while ~isempty(classqueue)
    mco = meta.class.fromName(classqueue{1});
    classqueue = classqueue(2:end);
    if isempty(mco)
        % MATLAB doesn't abstract Java class reflection,
        % so it will return an empty metaclass handle.        
        continue;
    end
    supers = mco.SuperClasses;    
    for i=1:numel(supers)
        name = supers{i}.Name;
        if ~ismember(name, mrolist)
            mrolist = [mrolist name]; %#ok<AGROW>
            classqueue = [classqueue name]; %#ok<AGROW>
        end
    end
end

if addvirtual
    if isnumeric(object) || ischar(object) || islogical(object)
        mrolist{end+1} = '_numeric';
    elseif isobject(object)
        mrolist{end+1} = '_object';
    elseif isjava(object)        
        mrolist{end+1} = '_java';
    % In case this gets used on Windows...
    elseif iscom(object)
        mrolist{end+1} = '_com';
    elseif isinterface(object)
        mrolist{end+1} = '_interface';
    end
end

if autosplit
    for i=1:numel(mrolist)
        classname = mrolist{i};
        dots = strfind(classname, '.');
        if isempty(dots)
            mrolist{i} = {'', classname};
        else
            d = dots(end);
            mrolist{i} = {classname(1:d-1), classname(d+1:end)};
        end
    end
end

% Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
% For full license details, see the LICENSE file.

