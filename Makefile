PYTHON ?= python2.6
MATLAB_SCRIPT ?= matlab
TMW_ROOT ?= $(shell ${MATLAB_SCRIPT} -e | grep MATLAB= | sed s/^MATLAB=//)

CFLAGS=$(shell ${PYTHON}-config --cflags)
CLIBS=$(shell ${PYTHON}-config --libs)
LDFLAGS=$(shell ${PYTHON}-config --ldflags) -L$(shell ${PYTHON}-config --prefix)/lib

MEXEXT ?= $(shell ${TMW_ROOT}/bin/mexext)

DEBUG ?= $(if $(wildcard .debug_1),1,0)
TARGET = pymex.${MEXEXT}

MEXFLAGS ?= 
MEXENV = CFLAGS="\$$CFLAGS ${CFLAGS}" CLIBS="\$$CLIBS ${CLIBS}" LDFLAGS="\$$LDFLAGS ${LDFLAGS}"
MEX = ${TMW_ROOT}/bin/mex 

all: ${TARGET}

${TARGET}: pymex.c sharedfuncs.c commands.c *module.c pymex.h .debug_${DEBUG}
	$(MEX) $(MEXFLAGS) $(MEXENV) -DPYMEX_DEBUG_FLAG=${DEBUG} pymex.c sharedfuncs.c *module.c

.debug_0:
	@echo "Debug disabled."
	@rm -f .debug_1
	@touch .debug_0

.debug_1:
	@echo "Debug enabled."
	@rm -f .debug_0
	@touch .debug_1

.PHONY: clean

clean:
	rm -f .debug_* pymex.mex*

