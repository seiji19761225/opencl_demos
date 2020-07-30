/*
 * wtime.c: wall clock timer
 * (c)2011-2016 Seiji Nishimura
 * $Id: wtime.h,v 1.1.1.1 2020/07/29 00:00:00 seiji Exp seiji $
 */

#ifndef __WTIME_H__
#define __WTIME_H__

#include <time.h>
#include <stdbool.h>

#ifdef  __WTIME_INTERNAL__
#define   WTIME_API
#else
#define   WTIME_API	extern
#endif

/* prototype */

#ifdef __cplusplus
extern "C" {
#endif

WTIME_API double wtime(void);

#ifdef __cplusplus
}
#endif
#undef    WTIME_API
#endif
