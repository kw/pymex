classdef voidptr < handle 
    %#ok<*MANU>
    %#ok<*INUSD>
    %#ok<*STOUT>
    properties (Hidden)
        pointer = uint64(0);
    end
    
    methods
        function B = subsref(A, S)  
            error('voidptr:subsref','Can''t subsref a void pointer');
        end
        
        function A = subsasgn(A, S, V) 
            error('voidptr:subsasgn','Can''t subsasgn a void pointer');
        end       
        
        function s = ptrstring(obj)
            s = sprintf('%04X',typecast(obj.pointer, 'uint32'));
        end
        
        function disp(obj) 
            disp(['(void*) 0x' obj.ptrstring]);
        end
        
        function n = numel(obj, varargin)
            n = 1;
        end
        
        function c = horzcat(varargin)
            c = cat(2, varargin{:});
        end
        
        function c = vertcat(varargin)
            c = cat(1, varargin{:});
        end
        
        function c = cat(dim, varargin)
            error('voidptr:cat','Can''t cat a void pointer');
        end
        
        function delete(obj)
                pymex('DELETE_OBJ', obj);
        end
    end
end

% Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
% For full license details, see the LICENSE file.

