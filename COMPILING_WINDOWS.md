# Compiling for Windows with Microsoft Tools #

This branch compiles on Windows using the free (as in beer) Microsoft Visual
Studio 2008 Express Edition. This seems the easiest compiler to use for building
64bit (it is hard to build mex64 files with mingw-64 on windows). 

You will need:

1. [Microsoft Visual Studio 2008 Express Edition](http://www.microsoft.com/express/downloads/)
2. [Microsoft Windows SDK](http://msdn.microsoft.com/en-us/windows/bb980924.aspx)
3. [Python 2.6 for Windows](http://python.org/download/)

The SDK is only required for [64 bit](http://tinyurl.com/3afmcka) (it extends VS
Express Edition with 64 bit tools). I used *Microsoft Windows SDK for Windows 7
and .NET Framework 3.5 Service Pack 1* (I think Framework 3.5 matches VS 2008
version). With these installed you should be able to run `mex -setup`. 

With mex setup correctly there is a `makeall.m` file which runs a one line mex
command to build the package. You may need to adjust the paths there to point to
your Python installation.

# Binaries #

If MATLAB's `mexext` command tells you `mexw64`, then you may be able to skip
the compilation and just drop
[pymex.mexw64](http://cloud.github.com/downloads/robince/pymex/pymex.mexw64)
into your pymex directory. You do still need the pymex distribution and to have
Python 2.6 installed. There is currently no difference between this and the
master branch other than small changes to enable compilation with Visual Studio,
so it shouldn't matter which branch you have out. Note that this binary was
compiled under MATLAB 2009a in Windows 7 with VS 2008 EE. It should work with
newer versions of MATLAB (but not older), not sure about other windows versions.
You may need the [VS 2008 Redistributable Package](http://tinyurl.com/6rm54q)


