/*
 * cl_util_internal.h: OpenCL utility
 * (c)2017-2021 Seiji Nishimura
 * $Id: cl_util_internal.h,v 1.1.1.3 2021/07/24 00:00:00 seiji Exp seiji $
 */

#ifndef __CL_UTIL_INTERNAL__
#define __CL_UTIL_INTERNAL__

#include "cl_util.h"

// configuration flag (undefine to turn off)
#define CL_UTIL_USE_STAT	1
#ifndef NDEBUG
#define CL_UTIL_DEBUG		1
#endif

#endif
