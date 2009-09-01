classdef BasePyObject < handle
  properties (Hidden)
      pointer = uint64(0);
  end
  properties (Access = protected, Transient)
      pytype
  end
  
  methods
      function delete(obj)
          pymex('DELETE_OBJ', obj);
      end
  end
end
