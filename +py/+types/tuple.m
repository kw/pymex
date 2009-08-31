classdef tuple < py.types.object
    methods
        function b = subsref(A, S)
            subs = S(1).subs;
            if strcmp(S(1).type, '{}')
                for s = 1:numel(subs)
                    subs{s} = int64(subs{s});
                end
            end
            S(1).subs = subs;
            b = subsref@py.types.object(A,S);
        end        
    end
end