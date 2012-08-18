# README for pymex #

pymex embeds a Python interpreter into MATLAB and provides support 
for interaction between them. It does not (currently) allow interaction 
with MATLAB through any arbitrary process - see the 'pymat' project for that. 

License is currently standard MIT/X11. See the LICENSE file.

If you're not sure how you got this, the git repository can be found at:
http://github.com/kw/pymex

See the Issues section near the bottom if you have problems, or send me
a message on GitHub, or maybe even send me an e-mail (kwatford@cise.ufl.edu)

# Requirements #

* MATLAB 2008b or higher. 
* Python 2.6. Probably needs shared rather than static.
  No Py3k support yet, but I'm trying to minimize porting
  issues. No NumPy for Py3k yet anyway.
* NumPy. Not a build requirement, but you'll definitely want it.
* (optional) nose, for running unit tests.
* If you're using windows, use the win32 branch, and see the
  COMPILING_WINDOWS.md file.

# Installation #

1. Clone or untar this somewhere.
2. Ensure that MATLAB's mex command is configured. Run `mex -setup` 
   if it isn't. (this can be done from within MATLAB or using the 
   mex script in MATLAB's bin dir)
3. Run `make` in that directory. If your configuration is not as 
   expected you can specify some of the environment variables, like 
   `make PYTHON=~/bin/python2.7`
4. If by some miracle that actually worked and you've got nose, try 
   `make test`. Note that you can't just run `nosetests` because the 
   necessary modules only exist within MATLAB. On some systems this 
   may fail rather hard -- see issue #1
5. Add the pymex directory to MATLAB's path. pymex will add itself
   to your Python path.

# Usage #

For some examples, see `python_example.py` and `matlab_example.py`.

`help pymex` might be helpful.

On the MATLAB side, `pyimport` and `pybuiltins` give us access to
Python modules and builtins. To import a module,

    pyimport numpy
    
or

    np = pyimport('numpy')
    
If you use the "command form" to import a submodule, it will change the
dots into underscores. I'd recommend using the expression form instead.

    pyimport mltypes.containers  % produces the variable "mltypes_containers"

Running `pybuiltins` with no arguments
tries to import all of Python's builtins, which is probably undesirable.
Provide string arguments to request specific builtins:

    pybuiltins list tuple dict
    
or

    [list, tuple, dict] = pybuiltins('list', 'tuple', 'dict');

Some shortcuts are provided in the 'py' package:

    py.tuple(42, 'spam', @plot)
    py.list(1, 2, 3)
    py.dict('spam', 'eggs', 'foo', 'bar')

(note that these shortcuts have different calling conventions
than the builtins. YMMV)

MATLAB has no keyword arguments, so to pass those to a python function, use the `kw` class:

    v = myfunc(a, b, kw('keyword1', 42, 'keyword2', py.None, ...))
    
You can provide multiple instances of `kw` if necessary.

To evaluate a python expression using the variables in the current MATLAB workspace, use `py.eval`:

    >> x = {'spam', 42, struct('foo','bar'), containers.Map}
    x = 
        'spam'    [42]    [1x1 struct]    [0x1 containers.Map]
    >> py.eval('[(type(a), a) for a in x]')
    ans = 
    [(<type 'str'>, 'spam'), (<class 'mltypes._builtins._numeric'>, 42), (<class 'mltypes._builtins.struct'>, 
     <struct at 0x7f12c84b7fd0>), (<class 'mltypes.containers.Map'>, <containers.Map at 0x7f12c84ba738>)]

Python objects tend to have a lot of attributes whose names begin with underscores.
That's not valid in MATLAB, but we can do it anyway using MATLAB's dynamic field access syntax:

    >> l = py.list(1,2,3)
    l = 
    [1, 2, 3]
    >> l.pop()           % normal attribute access
    ans = 
    3
    >> l.('__len__')()   % underscore attribute access
    ans = 
    2

This is ugly, but reinforces the idea that you probably shouldn't be accessing them.

Python functions and expressions do not automatically convert
outputs to MATLAB objects, even if they're just wrapped MATLAB
objects. Use the `unpy` method to coerce.

On the Python side:

    from matlab import cell, fprintf, plot, max

MATLAB functions (currently) coerce all arguments to MATLAB
types when possible, since presumably MATLAB functions don't
want Python inputs. To request multiple outputs, use the
'nargout' keyword:

    x = numpy.array([1, 2, 4, 0])
    val, ind = matlab.max(x, nargout=2) # ind is 1-based

# Wrappers #

Wrapper classes are provided for both sides of the river.
When an object needs to be wrapped, its method resolution
order is used to find a list of classes in order of
specificity. Then a specific hierarchy is checked for
matching classes, and the most specific match found wins.

On the MATLAB side, python classes are wrapped using the
`py.types` package, and the default type is `py.types.builtin.object`
(even for objects that aren't "object" instances). The only
other wrapper present at time of writing is `py.types.numpy.ndarray`.

On the Python side, MATLAB classes are wrapped using the
`mltypes` package. The base type is `mx.Array` (`mx` being
the embedded module representing MATLAB's libmx), but most
of the basic types have wrappers. `mro.m` provides the method
resolution order for an object, which includes some "virtual"
classes I've added whose names begin with underscores. The
prime example of this is the `_numeric` class, which all MATLAB
numeric arrays are subclasses of.

The `_object` class doesn't work particularly well for general MATLAB objects,
but see `mltypes.containers.Map` for an example of a wrapped MATLAB class:

    >> map = containers.Map;
    >> map('foo') = 'bar';
    >> pymap = py(map)
    pymap = 
      containers.Map handle
        Package: containers
	
      Properties:
            Count: 1
          KeyType: 'char'
        ValueType: 'any'
      Methods, Events, Superclasses
    >> pymap{'foo'}
    ans = 
    bar
    >> pymap{'spam'} = 'eggs';
    >> map('spam')
    ans =
    eggs
    >> 

# NumPy support #

The aforementioned `_numeric` class doesn't really do much
because normal MATLAB numeric arrays work so well with NumPy:

    >> x = py.asarray(magic(5)) % or pyimport numpy; numpy.asarray(...)
    x = 
    [[ 17.  24.   1.   8.  15.]
     [ 23.   5.   7.  14.  16.]
     [  4.   6.  13.  20.  22.]
     [ 10.  12.  19.  21.   3.]
     [ 11.  18.  25.   2.   9.]]
    >> x{0,0} = 42
    x = 
    [[ 42.  24.   1.   8.  15.]
     [ 23.   5.   7.  14.  16.]
     [  4.   6.  13.  20.  22.]
     [ 10.  12.  19.  21.   3.]
     [ 11.  18.  25.   2.   9.]]
    >> unpy(x)
    ans =
        42    24     1     8    15
        23     5     7    14    16
         4     6    13    20    22
        10    12    19    21     3
        11    18    25     2     9
    >> x.base
    ans = 
    [17 24 1 8 15;23 5 7 14 16;4 6 13 20 22;10 12 19 21 3;11 18 25 2 9]
    >> type(x.base)
    ans= 
    <class 'mltypes._builtins._numeric'>


# Issues #

Issue tracker is on GitHub. Please report problems there.
A few of note, so that you don't have to go looking for them:

* As mentioned, unit tests might not work. See Issue #1.
* MATLAB is not thread safe. See Issue #2.
* There is presently no support for complex or sparse matrices.
  See issues #5 and #6
* I presently have no way to generate an actual Python REPL in pymex.
  MATLAB does weird things with its stdin/stdout in its Desktop gui.
  If you start it up with `-nodesktop` then stdout works, but stdin
  doesn't seem to work properly (I've started Python's standard REPL
  but it doesn't seem to accept input). I've also tried simply starting
  IDLE, but it crashes somewhere within Tk/CoreFoundation (this is on my Mac)
  Maybe someone will have better luck, or just write a module based on MATLAB's
  `fprintf` and `input` functions.
