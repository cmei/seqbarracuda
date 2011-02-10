#######################################################
# Configuration for BarraCUDA project
#######################################################

# cu source files
CUFILES	= $(wildcard *.cu)
# c++ source files
CCFILES	= $(wildcard *.cpp)
# c source files
CFILES	= $(wildcard *.c)
# libraries dependency (zlib)
USERLIB = -lz 

#check for 64bit support and if yes build with it
ifeq ($(shell uname -m),x86_64)
COMMONFLAGS += -m64
endif

#check for pthread support and if yes build with it
ifneq ($(shell ldconfig -p | grep pthread),)
COMMONFLAGS += -DHAVE_PTHREAD
USERLIB += -lpthread
endif

#######################################################
# Default configuration (Autodetected)
#######################################################
#Cuda installation path (default)
CUDA_INSTALL_PATH := /usr/local/cuda
NV_ROOT_PATH := $(HOME)/NVIDIA_CUDA_SDK

# Basic Auto variable setup for SDK
SYSTEM        = $(subst ' ',_,$(shell uname | tr A-Z a-z))
PROJECT_PATH ?= .
EXECUTABLE   ?= $(lastword $(subst /, ,$(PWD)))
BINDIR       ?= $(SYSTEM)
ROOTOBJDIR   ?= $(SYSTEM)

ifeq ($(ConfigName),debug)
verbose = 1
dbg = 1
else 
verbose = 0
dbg = 0
endif

libcutil:
	cd ../..;make dbg=$(dbg) -C common
	
all : libcutil

include ../../common/common.mk
#######################################################
# End of configuration
#######################################################
