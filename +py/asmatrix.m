function a = asmatrix(varargin)
numpy = pyimport('numpy');
a = numpy.asmatrix(varargin{:});
