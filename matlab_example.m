% Example of a MATLAB function using Python
% Usage
%  >> matlab_example()
%  Class of x: double
%  Class of s: double
%  Class of c: py.types.numpy.ndarray
%  Python type of c: <type 'numpy.ndarray'>
% (Also produces a plot)
% See python_example.py for the inverse of this.
function matlab_example()
pyimport numpy
% We can only use this import syntax if the target is definitely
% not already a MATLAB function, since it does sneaky things
% with the function's workspace that the compiler can't predict.
% For safety, use: numpy = pyimport('numpy');

x = 0:0.1:5;
s = sin(x);
c = numpy.cos(x);
fprintf('Class of x: %s\n', class(x));
fprintf('Class of s: %s\n', class(s));
fprintf('Class of c: %s\n', class(c));
% MATLAB is more finicky about formatting than Python
fprintf('Python type of c: %s\n', char(type(c)));
% numpy knows when to convert to an ndarray, but
% MATLAB doesn't automatically convert the ndarray for us.
% So we'll have to ask it to convert. We ask explicitly for 
% type double here, but a plain 'unpy' would have done as well.
plot(x, s, x, double(c));

% Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
% For full license details, see the LICENSE file.
