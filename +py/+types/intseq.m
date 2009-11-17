% Integer-indexed sequence.
% Several common Python sequences can be indexed, but will throw a fit if you
% give them non-integer inputs. This is annoying, since all MATLAB numbers are
% double unless you go out of your way to cast them. This class makes things
% more convenient.
classdef intseq < py.types.builtin.object
    methods % Overrides
        function item = getitem(obj, varargin)                
            subs = py.types.intseq.fixkey(varargin{:});
            item = getitem@py.types.builtin.object(obj, subs);
        end
        function setitem(obj, val, varargin)
            subs = py.types.intseq.fixkey(varargin{:});
            setitem@py.types.builtin.object(obj, val, subs);
        end
    end
    
    methods (Static)
        function subs = fixkey(varargin)
            if numel(varargin) ~= 1
                error('intseq:KeyError', 'Must index on exactly one dimensions for this type of object.');
            end
            subs = varargin{1};
            if isnumeric(subs)
                if numel(subs) ~= 1
                    error('intseq:KeyError', ['This type does not support indexing over a sequence. ' ...
                        'You can use {a,b,c} as a shortcut for py.slice(a,b,c).']);
                else
                    return;
                end
            elseif iscell(subs)
                % either slice or the python object will complain if this is wrong
                subs = py.slice(subs{:});
            elseif isa(subs, 'py.types.builtin.object')
                % let python determine if it's convertible to an index
                return
            else
                error('intseq:KeyError', 'Don''t know how to index using a %s', class(subs))
            end            
        end        
    end
end
