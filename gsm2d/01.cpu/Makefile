#========================================================================
# makefile
# $Id: Makefile,v 1.1.1.1 2020/07/29 00:00:00 seiji Exp seiji $
#========================================================================
include ../../make.inc

.PHONY: default clean clobber

UTILS	= ../../utils
VPATH	= $(UTILS)

CFLAGS	+= -I$(UTILS)
LIBS	+= -lpixmap -lm
OBJS	= wtime.o
BIN	= gsm2d.exe
#------------------------------------------------------------------------
include config.mk
#------------------------------------------------------------------------
include input/$(DATA).dat
#........................................................................
PFLAGS	+= -DRR="$(RR)"
PFLAGS	+= -DDU="$(DU)"
PFLAGS	+= -DDV="$(DV)"
PFLAGS	+= -DF="$(F)"
PFLAGS	+= -DK="$(K)"
ifdef DELAY
PFLAGS	+= -DDELAY="$(DELAY)"
endif
ifdef U0
ifdef V0
PFLAGS	+= -DU0="$(U0)"
PFLAGS	+= -DV0="$(V0)"
endif
endif
ifeq ($(TYPE),float)
PFLAGS	+= -DUSE_FLOAT
endif
#------------------------------------------------------------------------
default: $(BIN)

$(BIN): gsm2d.c $(OBJS)
	$(CC) $(CFLAGS) $(PFLAGS) -o $@ $^ $(LIBS)

clean:
	@rm -f $(BIN) *.o *~ core*
	@rm -f *.ppm *.pgm *.pbm *.gz
clobber: clean
	@rm -f make.log run.log*
