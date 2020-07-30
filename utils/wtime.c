/*
 * wtime.c: wall clock timer
 * (c)2011-2016 Seiji Nishimura
 * $Id: wtime.c,v 1.1.1.1 2020/07/29 00:00:00 seiji Exp seiji $
 */

#include "wtime_internal.h"

//======================================================================
double wtime(void)
{				// wall clock timer
    struct timespec ts;

    if (clock_gettime(CLOCK_REALTIME, &ts) == -1)
	return -1.0;		// negative value means error.

    return (double) ts.tv_sec  +
	   (double) ts.tv_nsec * 1.E-9;
}
