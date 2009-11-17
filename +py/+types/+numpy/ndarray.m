classdef ndarray < py.types.builtin.object
    methods
        function s = size(obj, dim)
            s = double(getattr(obj, 'shape'));
            if numel(s) < 2
                s = [ones(1,2-numel(s)) s];
            end
            if nargin > 1
                s = s(dim);
            end
        end
        
        function b = transpose(a)
            b = methodcall(a,'transpose');
        end
        
        function b = ctranspose(a)
            % FIXME: Should probably throw a 'conjugate' in
            % here, but I'll ignore that until I've got the
            % complex number converters working.
            b = transpose(a);
        end
        
        function m = cat(dim, varargin)
            np = pyimport('numpy');
            dim = py.int(dim-1);            
            m = np.concatenate(varargin, dim);
        end
        
        function e = end(obj, k, n)
            s = size(obj);
            if k > numel(s)
                e = py.None;
            else
                e = s(k)-1;
            end
        end
        
    end
end
