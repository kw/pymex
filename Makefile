
PYTHON=$(shell which python)

PYTHONHOME=$(shell ${PYTHON} -c 'import sys; print(sys.prefix)')

PY_VER = $(shell ${PYTHON} -c 'import sys; print("%d.%d" % sys.version_info[0:2])')

PY_INCLUDE=${PYTHONHOME}/include/python${PY_VER}
PY_LIB=${PYTHONHOME}/lib
NUMPY_INCLUDE ?= $(shell ${PYTHON} -c 'import numpy; print(numpy.get_include());')
PY_FLAGS=-I${PY_INCLUDE} -I${NUMPY_INCLUDE} -L${PY_LIB} -lpython${PY_VER}

MATLAB ?= $(shell matlab -e | grep MATLAB= | sed s/^MATLAB=//)

DEBUG ?= $(if $(wildcard .debug_1),1,0)

TARGET = $(word 1, $(wildcard pymex.mex*) pymex.mex)

MEXFLAGS=-g -argcheck
MEX=${MATLAB}/bin/mex

all: ${TARGET}

${TARGET}: pymex.c pymex_static.c pymex.def.c .debug_${DEBUG}
	$(MEX) $(MEXFLAGS) ${PY_FLAGS} -DPYMEX_DEBUG_FLAG=${DEBUG} pymex.c

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
