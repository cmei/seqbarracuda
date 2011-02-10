# CUDA SDK 10 Linux Version 2.20.0410.1615
# Modified for BarraCUDA 0.5 beta release
ifeq ($(emu), 1)
  PROJECTS := $(shell find projects -name Makefile | xargs grep -L 'USEDRVAPI')
else
  PROJECTS := $(shell find projects -name Makefile)
endif

%.ph_build : lib/libcutil.so
	make -C $(dir $*) $(MAKECMDGOALS)

%.ph_clean : 
	make -C $(dir $*) clean $(USE_DEVICE)

%.ph_clobber :
	make -C $(dir $*) clobber $(USE_DEVICE)

all:  $(addsuffix .ph_build,$(PROJECTS))
	@echo "Finished building all"

lib/libcutil.so:
	@make -C common

tidy:
	@find * | egrep "#" | xargs rm -f
	@find * | egrep "\~" | xargs rm -f

clean: tidy $(addsuffix .ph_clean,$(PROJECTS))
	@make -C common clean

clobber: clean $(addsuffix .ph_clobber,$(PROJECTS))
	@make -C common clobber
