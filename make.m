myroot = getenv('MYROOT');
incdir = ['-I' fullfile(myroot, 'include/python2.6')];
libdir = ['-L' fullfile(myroot, 'lib')];
lib = '-lpython2.6';
code = 'pymex.c';
mex(incdir, libdir, lib, code);

