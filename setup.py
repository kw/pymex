from distutils.core import setup, Extension
import os

#FIXME - automate this
MATLAB = '/Applications/MATLAB_R2008bSV.app/'
MATLAB_INC = os.path.join(MATLAB, 'extern', 'include')
MATLAB_ARCH = 'maci'
MATLAB_LIB = os.path.join(MATLAB, 'bin', MATLAB_ARCH)

pymx = Extension('pymx', ['pymxmodule.c'],
                  define_macros=[('MX_COMPAT_32',1)],
                  include_dirs=[MATLAB_INC],
                  library_dirs=[MATLAB_LIB],
                  runtime_library_dirs=[MATLAB_LIB],
                  libraries=['mx'])
                  

setup(name='pymx',
      version='0.0',
      ext_modules=[pymx])
