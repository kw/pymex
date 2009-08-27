function imp = import(name)
md = pymex('GET_MODULE_DICT');
try
    imp = md{name};
catch
    imp = pymex('IMPORT', name);
end
