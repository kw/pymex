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
	    if mod(nargin,2)
	      error('kw:needspairs', 'Input must be in pairs.');
	    elseif nargin == 0
	      kwargs(1) = [];
            end	      
	    if nargin == 2	      
	      kwargs.keyword = varargin{1};
	      kwargs.value = varargin{2};
	      assert(ischar(kwargs.keyword));
	    else
	      for i = 1:2:nargin
		arg = arg + 1;		
		kwargs(arg) = kw(varargin{i:i+1});
	      end
	    end
        end
        
        function d = dict(kwargs)
            dict = pybuiltins('dict');
            d = dict();
            for i=1:numel(kwargs)
                d{kwargs(i).keyword} = kwargs(i).value;
            end
        end
    end    
end
 
% Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
% For full license details, see the LICENSE file.

