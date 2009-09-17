% Integer-indexed sequence.
% Several common Python sequences can be indexed, but will throw a fit if you
% give them non-integer inputs. This is annoying, since all MATLAB numbers are
% double unless you go out of your way to cast them. This class makes things
% more convenient.
classdef intseq < pytypes.object
    methods % Overrides
        function item = getitem(obj, key)
            item = getitem@pytypes.object(obj, fixkey(key));
        end
        function setitem(obj, key, val)
            setitem@pytypes.object(obj, fixkey(key), val);
        end
    end
    
    methods (Access = private)
        function key = fixkey(key)
            if isnumeric(key) && ~isint(key)
                key = int64(key);
                return;
            end
            if isa(key, 'pytypes.object')
                [int long slice None] = pybuiltins('int','long','slice','None');
                if isinstance(key, slice)
                    % Adjust slice components
                    components = {'start','step','stop'};
                    for c = components
                        st = getattr(slice, c);
                        if st ~= None && ~(st.type == int || st.type == long)
                            setattr(slice, c, long(st));
                        end
                    end
                else
                    try
                        numpy = pyimport('numpy');
                        key = numpy.ndarray(key);
                        key = key.astype(long);
                    catch %#ok
                        % ignore error and pass along the key
                    end
                end
            end
        end
    end
end