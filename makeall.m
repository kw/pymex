

%mex -IC:\Python26\include pymex.cpp
%mex -IC:\Python26\include sharedfuncs.cpp
% mex -IC:\Python26\include engmodule.cpp
% mex -IC:\Python26\include mexmodule.cpp
%mex -IC:\Python26\include mxmodule.cpp
%mex -IC:\Python26\include matmodule.cpp


mex -v -IC:\Python26\include pymex.cpp sharedfuncs.cpp engmodule.cpp matmodule.cpp mexmodule.cpp mxmodule.cpp C:\Python26\libs\python26.lib
