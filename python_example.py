'''
Example of a python module using pymex.
You must be in MATLAB to run it.
Usage:
 >> pyimport python_example
 >> python_example.run()
 Type of x: <type 'numpy.ndarray'>
 Type of s: <type 'numpy.ndarray'>
 Type of c: <class 'mltypes._numeric'>
 mxclass of c: double
(Also produces a plot)
See matlab_example.m for the inverse of this.
'''
# Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
# For full license details, see the LICENSE file.

from numpy import arange, sin
from matlab import fprintf, cos, plot, _class
# class is a python keyword, so to import the MATLAB 'class'
# we can prepend an underscore. 

def run():
    x = arange(0,5.1,0.1)
    s = sin(x)
    c = cos(x)
    # Python's normal printing capability doesn't work
    # unless you're using MATLAB without the GUI.
    # Use MATLAB's printing and reading functions.
    fprintf("Type of x: %r\n" % type(x))
    fprintf("Type of s: %r\n" % type(s))
    fprintf("Type of c: %r\n" % type(c))
    fprintf("mxclass of c: %s\n" % _class(c))
    # Currently pymex will try to coerce all function
    # arguments to MATLAB types if it can. This should
    # probably be configurable using keyword args...
    plot(x, s, x, c)

