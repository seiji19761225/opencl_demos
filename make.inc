#========================================================================
# make.inc
# $Id: make.inc,v 1.1.1.2 2021/07/10 00:00:00 seiji Exp seiji $
#========================================================================
.SUFFIXES: .c .o

CC	= gcc
CFLAGS	= -O3 -fopenmp -std=gnu99

PFLAGS	= -DCL_TARGET_OPENCL_VERSION=120
LIBS	=
LIBOCL	= -lOpenCL
#------------------------------------------------------------------------
.c.o:
	$(CC) $(CFLAGS) $(PFLAGS) -c -o $@ $<
