function varargout = import(name)
md = pymex('GET_MODULE_DICT');
try
    imp = md{name};
catch %#ok<CTCH>
    imp = pymex('IMPORT', name);
end
if nargout == 0
    assignin('caller', name, imp);
else
    varargout{1} = imp;
end
