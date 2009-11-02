% Integer-indexed sequence.
% Several common Python sequences can be indexed, but will throw a fit if you
% give them non-integer inputs. This is annoying, since all MATLAB numbers are
% double unless you go out of your way to cast them. This class makes things
% more convenient.
classdef intseq < pytypes.builtin.object
    methods % Overrides
        function item = getitem(obj, varargin)
            subs = pytypes.intseq.fixkey(varargin{:});
            item = getitem@pytypes.builtin.object(obj, subs{:});
        end
        function setitem(obj, val, varargin)
            subs = pytypes.intseq.fixkey(varargin{:});
            setitem@pytypes.builtin.object(obj, val, subs{:});
        end
    end
    
    methods (Static)
        function subs = fixkey(varargin)
            subs = cell(size(varargin));            
            for i=1:numel(varargin)
                key = varargin{i};
                if isempty(key)
                    %ignore
                elseif isnumeric(key)
                    key = py.int(key);                    
                elseif islogical(key)
                    key = py.bool(key);
                elseif iscell(key)
                    key = pytypes.intseq.fixkey(key{:});
                    key = py.slice(key{:});
                elseif ~isa(key,'pytypes.object')
                    error('IntegerIndexedSequence:BadKey','Don''t know what to do with key of type %s', class(key));
                end
                subs{i} = key;
            end            
        end
    end
end
