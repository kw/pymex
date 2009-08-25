function imp = import(name)
md = pymex(py.Interface.GET_MODULE_DICT);
try
    imp = md{name};
catch
    imp = pymex(py.Interface.IMPORT, name);
end
