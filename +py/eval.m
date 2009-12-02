function val = eval(evalstr)
try
    globals = pybuiltins('globals');
    global_dict = globals();
catch    
    global_dict = py.dict();    
end
locals = py.dict();
names = evalin('caller', 'who');
for n = 1:numel(names)
    locals{names{n}} = evalin('caller', names{n});
end
pyeval = pybuiltins('eval');
val = pyeval(evalstr, global_dict, locals);
