#========================================================================
# makefile
# $Id: Makefile,v 1.1.1.1 2020/07/30 00:00:00 seiji Exp seiji $
#========================================================================
include ../../make.inc

.PHONY: default clean clobber

UTILS	= ../../utils
VPATH	= $(UTILS)

CFLAGS	+= -I$(UTILS)
LIBS	+= -lpixmap -lm
OBJS	= wtime.o
BIN	= sobel.exe
#------------------------------------------------------------------------
include config.mk
#------------------------------------------------------------------------
PFLAGS	+= -DINPUT_FILE="\"$(INPUT)\""
#------------------------------------------------------------------------
default: $(BIN)

$(BIN): sobel.c $(OBJS)
	$(CC) $(CFLAGS) $(PFLAGS) -o $@ $^ $(LIBS)

clean:
	@rm -f $(BIN) *.o *~ core*
	@rm -f *.ppm *.pgm *.pbm *.gz
clobber: clean
	@rm -f make.log run.log*
