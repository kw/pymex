classdef list < py.types.object
    methods
        function b = subsref(A, S)
            subs = S(1).subs;
            if strcmp(S(1).type, '{}')
                for s = 1:numel(subs)
                    if isfloat(subs{s})
                        subs{s} = int64(subs{s});
                    end
                end
            end
            S(1).subs = subs;
            b = subsref@py.types.object(A,S);
        end
        
        function A = subsasgn(A, S, val)                            
            if numel(S) == 1
                subs = S.subs;
                if strcmp(S(1).type, '{}')
                    for s = 1:numel(subs)
                        if isfloat(subs{s})
                            subs{s} = int64(subs{s});
                        end
                    end
                end
                S.subs = subs;                
            end
            A = subsasgn@py.types.object(A,S,val);
        end
    end
end