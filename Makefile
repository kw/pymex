PYDIR ?= C:/Python26
CFLAGS= -I$(shell cygpath -m ${PYDIR}/include)
CLIBS= $(shell cygpath -m ${PYDIR}/libs/libpython26.a)

BUILDBRANCH = $(shell git branch --no-color | sed -e '/^[^*]/d' -e 's/* \(.*\)/\1/')
ifeq ($(BUILDBRANCH),)
 BUILDBRANCH = unknown
endif
BUILDTAG = $(shell git describe)
ifeq ($(BUILDTAG),)
 BUILDTAG = unknown
endif
BUILDNAME = $(BUILDBRANCH)/$(BUILDTAG)

MEXEXT ?= $(shell mexext.bat)

DEBUG ?= $(if $(wildcard .debug_1),1,0)
TARGET = pymex.${MEXEXT}

MEXFLAGS ?= 
MEX = mex.bat

all: ${TARGET}

${TARGET}: pymex.c sharedfuncs.c commands.c *module.c pymex.h .debug_${DEBUG}
	@echo building $(BUILDNAME)
	$(MEX) $(MEXFLAGS) $(CFLAGS) \
	-DPYMEX_DEBUG_FLAG=$(DEBUG) \
	-DPYMEX_BUILD="$(BUILDNAME)" \
	pymex.c sharedfuncs.c *module.c $(CLIBS)

.debug_0:
	@echo "Debug disabled."
	@rm -f .debug_1
	@touch .debug_0

.debug_1:
	@echo "Debug enabled."
	@rm -f .debug_0
	@touch .debug_1

test: ${TARGET} *.py
	${MATLAB_SCRIPT} -nojvm -nodisplay \
	-r "pyimport nose; exit(unpy(~nose.run()));"

.PHONY: clean

clean:
	rm -f .debug_* pymex.mex*

