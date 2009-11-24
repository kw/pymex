% out = any2str(arg)
% Produce a string representation of any MATLAB value. 
% Most languages that even hint at object orientation have this
% sort of facility (Python has at least two...), but MATLAB instead
% provides several mutually incompatible methods. There is a "disp"
% command that objects are generally expected to provide (or use
% the default), but there doesn't seem to be a an "sprintf"
% equivalent in case you want to use the string for something. But
% you *can* trap the output with evalc. mat2str's output is usually
% better when we can get it, so we try that first.
function out = any2str(arg)
try
  out = mat2str(arg);
catch %#ok
  try
    out = evalc('disp(arg)');
  catch %#ok
    out = sprintf('<%s instance (undisplayable)>', class(arg));
  end
end

% Copyright (c) 2009 Ken Watford (kwatford@cise.ufl.edu)
% For full license details, see the LICENSE file.

