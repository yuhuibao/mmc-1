ROOTDIR = ..
BINARY=mmc

FILES=xorshift128p_rand simpmesh tettracing mcx_utils tictoc mmc cjson/cJSON mmc_host highordermesh mmcx_core mmcx_host

ifeq ($(findstring Darwin,$(PLATFORM)), Darwin)
  EXTRALIB:=-static-libgcc -lgcc_eh
  MEXLINKOPT:=$(EXTRALIB)
endif

DOXYCFG=mmcdoxy.cfg

USERCCFLAGS=-DUSE_OS_TIMER

DUMMY:=$(shell mkdir -p built/cjson)

include $(ROOTDIR)/commons/Makefile_common.mk

