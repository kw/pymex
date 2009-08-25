classdef kw
    properties
        keyword
        value
    end
    
    methods
        function kwargs = kw(varargin)
            arg = 0;
            for i = 1:2:nargin
                arg = arg + 1;
                kwargs(arg).keyword = varargin{i};
                kwargs(arg).value = varargin{i+1};
            end
        end
        
        function d = dict(kwargs)
            d = pymex(py.Interface.DICT_FROM_KW, kwargs);
        end
    end
    
    
end
                