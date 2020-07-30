#========================================================================
# makefile
# $Id: Makefile,v 1.1.1.1 2020/07/29 00:00:00 seiji Exp seiji $
#========================================================================
include ../../make.inc

.PHONY: default clean clobber

UTILS	= ../../utils
VPATH	= $(UTILS)

CFLAGS	+= -I$(UTILS)
LIBS	+= -lm -lpixmap
OBJS	= wtime.o
BIN	= ao.exe
#------------------------------------------------------------------------
include config.mk
#........................................................................
ifeq ($(TYPE),float)
PFLAGS	+= -DUSE_FLOAT
endif

ifdef DELAY
PFLAGS	+= -DDELAY="$(DELAY)"
endif

PFLAGS	+= -DWIDTH="$(WIDTH)" -DHEIGHT="$(HEIGHT)"
PFLAGS	+= -DNSUBSAMPLES="$(SUB_SAMPLE)"
PFLAGS	+= -DNAO_SAMPLES="$(AO_SAMPLE)"
PFLAGS	+= -DNFRAMES="$(FRAME)"
#------------------------------------------------------------------------
default: $(BIN)

$(BIN): ao.c $(OBJS)
	$(CC) $(CFLAGS) $(PFLAGS) -o $@ $^ $(LIBS)

clean:
	@rm -f $(BIN) *.o *~ core*
	@rm -f *.ppm *.pgm *.pbm *.gz
clobber: clean
	@rm -f make.log run.log*
