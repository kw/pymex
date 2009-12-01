# README for pymex #

pymex embeds a Python interpreter into MATLAB and provides support for interaction between them. 
It does not (currently) allow interaction with MATLAB through any arbitrary process - see the 
'pymat' project for that. 

# Requirements #

* MATLAB 2008b or higher. 
* Python 2.6. Probably needs shared rather than static.
* NumPy. Not a build requirement, but you'll definitely want it.
* (optional) nose, for running unit tests.
* Not Windows. This isn't a zealotry thing, I just don't normally have a Windows machine
  with all of the above available, so I doubt the Makefile will work right. There's a 
  chance it might work with cygwin, but it will probably need modification to work properly.

# Installation #

1. Clone or untar this somewhere.
2. Ensure that MATLAB's mex command is configured. Run `mex -setup` if it isn't. 
   (this can be done from within MATLAB or using the mex script in MATLAB's bin dir)
3. Run `make` in that directory. If your configuration is not as expected you can
   specify some of the environment variables, like `make PYTHON=~/bin/python2.7`
3. If by some miracle that actually worked and you've got nose, try `make test`. 
   Note that you can't just run `nosetests` because the necessary modules only exist
   within MATLAB. On some systems this may fail rather hard -- see issue #1