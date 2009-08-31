classdef object < handle    
  properties (Hidden)
      pointer
  end
  properties (Access = private, Transient)
      pytype
  end
  
  methods
      function obj = object(inobj)
          obj.pointer = uint64(0);
      end
      
      function delete(obj)
          pymex('DELETE_OBJ', obj);
      end
      
      function objdir = dir(obj)
          objdir = pymex('DIR', obj);
      end
      
      function tf = is(obj1, obj2)
          tf = pymex('IS', obj1, obj2);
      end
      
      function c = plus(a, b)
          c = pymex('ADD', a, b);
      end
      
      function c = minus(a, b)
          c = pymex('SUBTRACT', a, b);
      end
      
      function c = mtimes(a, b)
          c = pymex('MULTIPLY', a, b);
      end
      
      function c = mrdivide(a, b)
          c = pymex('DIVIDE', a, b);
      end
      
      function c = rem(a, b)
          c = pymex('REM', a, b);
      end
      
      function c = mod(a, b)
          c = pymex('MOD', a, b);
      end
      
      function c = bitand(a, b)
          c = pymex('BITAND', a, b);
      end
      
      function c = bitor(a, b)
          c = pymex('BITOR', a, b);
      end
      
      function c = bitxor(a, b)
          c = pymex('BITXOR', a, b);
      end
      
      function tf = logical(obj)
          tf = pymex('TO_BOOL', obj);
      end
      
      function attr = getattr(obj, attrname)
          attr = pymex('GET_ATTR', obj, attrname);
      end
      
      function item = getitem(obj, key)
          item = pymex('GET_ITEM', obj, key);
      end
      
      function setattr(obj, attrname, val)
          pymex('SET_ATTR', obj, attrname, val);
      end
      
      function setitem(obj, key, val)
          pymex('SET_ITEM', obj, key, val);
      end
      
      function c = char(obj)
          if obj.pointer == uint64(0)
              c = '<null pointer>';
          else
              c = pymex('TO_STR', obj);
          end
      end
      
      function t = type(obj)
          if isempty(obj.pytype)              
              if obj.pointer == uint64(0)
                  t = py.types.object();
              else
                  t = pymex('GET_TYPE', obj);
              end
              obj.pytype = t;
          else
              t = obj.pytype;
          end
      end      
      
      function r = call(obj, varargin)
          iskw = cellfun(@(o) isa(o, 'py.kw'), varargin);
          kwargs = [varargin{iskw}];
          args = varargin(~iskw);
          r = pymex('CALL', obj, args, kwargs);
      end
      
      function r = abs(obj)
          r = subsref(obj, substruct('.', '__abs__', '()', {}));
      end
      
      function r = doc(obj)
          r = subsref(obj, substruct('.', '__doc__'));
      end
      
      function r = hash(obj)
          r = subsref(obj, substruct('.', '__hash__', '()', {}));
      end
      
      function r = hex(obj)
          r = subsref(obj, substruct('.', '__hex__', '()', {}));
      end
      
      function r = invert(obj)
          r = subsref(obj, substruct('.', '__invert__', '()', {}));
      end
      
      function r = oct(obj)
          r = subsref(obj, substruct('.', '__oct__', '()', {}));
      end
      
      function r = str(obj)
          r = subsref(obj, substruct('.', '__str__', '()', {}));
      end
      
      function r = len(obj)
          r = subsref(obj, substruct('.', '__len__', '()', {}));
      end
      
      function r = name(obj)
          r = subsref(obj, substruct('.', '__name__'));
      end
      
      function disp(obj)
          str = char(obj);
          toolong = 80*24;
          toomany = 40;
          newlines = strfind(str, char(10));
          typename = ['pymex ' char(name(type(obj)))];
          
          if numel(newlines) > toomany
              fprintf('%s:\n%s\n...<truncated: too many lines>...\n', typename, str(1:newlines(toomany)-1));
          elseif numel(str) > toolong
              fprintf('%s:\n%s\n...<truncated: too long>...\n', typename, str(1:toolong));
          elseif numel(newlines) == 0 && numel(str) < 80
              fprintf('%s: %s\n', typename, str);
          else
              fprintf('%s:\n%s\n', typename, str);             
          end
              
      end
      
      function n = numel(obj, varargin)
          n = 1;
      end
      
      function tf = iscallable(obj)
          tf = pymex('IS_CALLABLE', obj);
      end
      
      function tf = isinstance(obj, pytype)
          tf = pymex('IS_INSTANCE', obj, pytype);
      end
      
      function pstruct = saveobj(obj)
          pstruct = struct('pickled',false,'string','');
          try
              dumps = getattr(py.import('pickle'), 'dumps');
              pstruct.string = char(call(dumps, obj));
              pstruct.pickled = true;
          catch %#ok<CTCH>
              warning('pyobj:pickle', 'could not pickle object');
          end
      end      
      
      function varargout = subsref(obj, S)
          out = obj;
          subs = S(1).subs;
          switch S(1).type
              case '.'
                  out = getattr(out, subs);
              case '()'
                  out = call(out, subs{:});
              case '{}'
                  for s = 1:numel(subs)
                      if isequal(subs{s}, ':')
                          subs{s} = py.slice([],[],[]);
                      end
                  end                   
                  if numel(subs) > 1
                      out = getitem(out, subs);
                  else
                      out = getitem(out, subs{1});
                  end
              otherwise
                  error('wtf is "%s" doing in a substruct?', S(1).type);                  
          end
          if numel(S) > 1
              out = subsref(out, S(2:end));
          end          
          if ~(nargout == 0 && is(py.None, out))
              varargout{1} = out;
          end                  
      end
               
      function obj = subsasgn(obj, S, val)
          if numel(S) > 1
              preS = S(1:end-1);
              S = S(end);
              if strcmp(S.type, '()')
                  error('PyObject:BadAssign', 'Invalid lvalue. Can''t assign to expression ending in ().');
              end
              endobj = subsref(obj, preS);
              subsasgn(endobj, S, val);
          else
              subs = S.subs;
              switch S.type
                  case '.'
                      setattr(obj, subs, val);
                  case '{}'
                      for s = 1:numel(subs)
                          if isequal(subs{s}, ':')
                              subs{s} = py.slice([],[],[]);
                          end
                      end
                      if numel(subs) > 1
                          setitem(obj, subs, val);
                      else
                          setitem(obj, subs{1}, val);
                      end
                  otherwise
                      error('wtf is "%s" doing in a substruct?', S.type);
              end
          end
      end
      
      function n = colon(a, b, c)
          if nargin == 2
              n = py.slice(a, b, []);
          else
              n = py.slice(a, c, b);
          end
      end
      
      function n = end(obj, k, n) %#ok<INUSD>
          if k > 1
              warning('pymex:end','End of dimension %d requested, but generic objects don''t have that many.',k);
          end
          n = len(obj)-int64(1);
      end
  end
  
  methods (Static)
      function pyobj = loadobj(pstruct)
          if ~pstruct.pickled
              pyobj = py.None;
          else
              try
                  loads = getattr(py.import('pickle'), 'loads');
                  pyobj = call(loads, pstruct.string);
              catch %#ok<CTCH>
                  pyobj = py.None;
                  warning('pyobj:unpickle', 'could not load pickled object');
              end
          end
      end
  end

end

    
    
