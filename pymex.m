function varargout = pymex(varargin) %#ok<STOUT>
% PYMEX is Python embedded in MATLAB.
% More specifically, the pymex mex file contains various "kernel" commands
% that are used by various aspects of the system to communicate with the
% Python interpreter. As a user, you should probably not need to call it
% directly. Instead, you should interact with it via:
% * The pybuiltins function and pyimport function
% * The py and unpy type converters
% * The helper functions defined in the py package, 
%   such as py.asmatrix or py.None
% * The wrapper classes defined in the py.types package.
%
% If you do need to use pymex directly, run
%   c = pymex
% to get a listing of the available kernel calls, and
%   pymex help CMDNAME
% to see help for a specific command
% To modify the list of commands, edit commands.c and recompile.

error('pymex:NotBuilt','You must first build pymex.%s using the makefile.', mexext);
