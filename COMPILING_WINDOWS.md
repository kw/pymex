# Compiling for Windows #

This is how I managed to compile pymex in Windows. There's probably 
a better way, but I try not to use Windows that much myself. Maybe
I'll try one of Microsoft's compilers later, but MATLAB's lcc doesn't
seem to like Python's headers.

You will need:

1. [cygwin](http://www.cygwin.com/)
2. [MinGW](http://www.mingw.org/)
3. Python 2.6 for Windows (not just the cygwin version)
4. [gnumex](http://gnumex.sourceforge.net/)

Tell gnumex where both cygwin and MinGW are, but configure it for a MinGW build.
If you haven't already, check out pymex and make sure you're on the win32 branch
(the Makefile is different). If you don't have git already, consider using your
cygwin installer to get it, otherwise just make sure to get the right archive
from GitHub. 

Now, in Cygwin's bash, travel to the pymex directory and run `make`. If your Python26
directory is somewhere other than C:\Python26, you'll need to tell make about it:
`make PYDIR=D:/MyStuff/Python26`. Use normal slashes, not backslashes. 

# Notes #

Why mingw *and* cygwin? 
Cygwin has useful tools like `make` and `git`, but for some reason gnumex does not
produce an appropriate set of mex options for cygwin's gcc (at least for me). 
Compiling with MinGW and the standard Python for Windows also avoids depdendence on
Cygwin. The compiled mexw32 or mexw64 file should only require Python to work.
Of course, if you have installed the right msys ports you can probably do without cygwin.

# Binaries #

If MATLAB's `mexext` command tells you `mexw32`, then you may be able to skip the compilation
and just drop [pymex.mexw32](http://cloud.github.com/downloads/kw/pymex/pymex.mexw32) into your
pymex directory. You do still need the pymex distribution and to have Python 2.6 installed. There
is currently no difference between this and the master branch other than compilation, so it shouldn't
matter which branch you have out. Note that this binary was compiled under MATLAB 2008b in Windows 7
with MinGW's gcc. I have no idea whether it will work on other configurations. 

There is presently no `mexw64` binary available. Sorry.