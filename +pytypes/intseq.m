% Integer-indexed sequence.
% Several common Python sequences can be indexed, but will throw a fit if you
% give them non-integer inputs. This is annoying, since all MATLAB numbers are
% double unless you go out of your way to cast them. This class makes things
% more convenient.
classdef intseq < pytypes.object
    methods % Overrides
        function item = getitem(obj, key)
            item = getitem@pytypes.object(obj, pytypes.intseq.fixkey(key));
        end
        function setitem(obj, key, val)
            setitem@pytypes.object(obj, pytypes.intseq.fixkey(key), val);
        end
    end
    
    methods (Static)
        function key = fixkey(key)
            if isinteger(key)
                % ok!
            elseif isnumeric(key) && ~isinteger(key)
                key = int64(key);
            elseif isa(key, 'pytypes.object')
                [int long slice None] = pybuiltins('int','long','slice','None');
                if isinstance(key, slice)
                    % Adjust slice components
                    components = {'start','step','stop'};
                    for c = components
                        st = getattr(key, c{1});
                        t = type(st);
                        if is(st,None) && ~(is(t,int) || is(t,long))
                            try
                                setattr(key, c{1}, long(st));
                            catch                                
                            end
                        end
                    end
                else
                    try
                        numpy = pyimport('numpy');
                        key = methodcall(numpy, 'array', key);
                        key = methodcall(key, 'astype', 'int64');
                    catch %#ok
                        % ignore error and pass along the key
                    end
                end              
            end
        end
    end
end