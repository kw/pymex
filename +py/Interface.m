% py.Interface
% Currently the primary purpose of this class is to contain the constants
% used to communicate with the pymex module. It may at some point get some
% use. For now, the only thing its instances do is keep the pymex module
% from clearing, but any non-null py.Object will do that as well.
classdef Interface < handle
    properties (Constant, Hidden)
        MEXLOCK = 1;
        MEXUNLOCK = 2;
        DELETE_OBJ = 3;
        GET_GLOBALS = 4;
        GET_LOCALS = 5;
        GET_BUILTINS = 6;
        GET_FRAME = 7;
        IMPORT = 8;
        TO_STR = 9;
        DIR = 10;
        GET_ATTR = 11;
        GET_TYPE = 12;
        SCALAR_TO_PYOBJ = 13;
        TO_LIST = 14;
        TO_TUPLE = 15;
        CALL = 16;
        DICT_FROM_KW = 17;
        GET_ITEM = 18;
        IS_CALLABLE = 19;
        SET_ATTR = 20;
        SET_ITEM = 21;
        GET_MODULE_DICT = 22;
        IS_INSTANCE = 23;
    end
    
    methods
        function P = Interface()
            pymex(P.MEXLOCK);
        end
        
        function delete(P)
            pymex(P.MEXUNLOCK);
        end
    end
end

