function a = asarray(varargin)
numpy = pyimport('numpy');
a = numpy.asarray(varargin{:});
