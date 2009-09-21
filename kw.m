% Class kw: Python Keywords
% Python allows function calls to include keyword arguments as well as
% positional arguments. In some cases arguments *must* be supplied as keywords.
% MATLAB has no similar syntax, so some sort of marker is needed to indicate
% keyword arguments to be provided to Python calls, like so:
% x = pyfunc(a, kw('b', 'bee'), c, kw('d', 'dee', 'e', 'iii'))
% The kw() constructor takes key/value pairs and produces an object array. 
% When MATLAB tries to call a python function using the 'call' method (or the ()
% syntax), any objects of type kw will be concatenated, converted to a python
% dict, and passed in as the **kwargs parameter. 
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
                kwargs(arg).keyword = varargin{i}; %#ok<AGROW>
                kwargs(arg).value = varargin{i+1}; %#ok<AGROW>
            end
            if arg==0
                kwargs(1) = [];
            end
        end
        
        function d = dict(kwargs)
            dict = pybuiltins('dict');
            d = dict();
            for i=1:numel(kwargs)
                d{kwargs(i).keyword} = kwargs{i}.value;
            end
        end
    end
    
    
end
 