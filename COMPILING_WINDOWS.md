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