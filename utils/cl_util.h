/*
 * cl_util.h: utilities for OpenCL parallel processing
 * (c)2017 Seiji Nishimura
 * $Id: cl_util.h,v 1.1.1.1 2020/07/29 00:00:00 seiji Exp seiji $
 */

#ifndef __CL_UTIL_H__
#define __CL_UTIL_H__

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#ifdef __CL_UTIL_INTERNAL__
#define CL_UTIL_API
#else
#define CL_UTIL_API	extern
#endif

// for memory size alignment
#if 1
// This is fast, but "unit" is limited to powers of two.
#define ROUND_UP(n,unit)	(((int) (n)+((unit)-1))&~((unit)-1))
#else
#define ROUND_UP(n,unit)	((((int) (n)+((unit)-1))/(unit))*(unit))
#endif

typedef struct {
    cl_device_id     device ;
    cl_program       program;
    cl_command_queue queue  ;
    cl_context       context;
} cl_obj_t;

// macro functions to handle cl_obj_t
#define cl_init(obj,dev,kernel,options) \
{ \
    cl_int stat; \
    if ((stat = cl_Init(dev, \
			&(obj)->device, &(obj)->program, \
			&(obj)->queue , &(obj)->context, kernel, options)) != CL_SUCCESS) \
	cl_CheckStatus("cl_Init", stat); \
}
#define cl_set_local_size(obj,ndim,global_size,local_size) \
{ \
    cl_int stat; \
    if ((stat = cl_SetThreadLocalSize((obj)->device, ndim, \
					global_size, local_size)) != CL_SUCCESS) \
	cl_CheckStatus("cl_SetThreadLocalSize", stat); \
}
#define cl_fin(obj)		cl_Fin(&(obj)->device, &(obj)->program, &(obj)->queue, &(obj)->context)
#define cl_query_device(obj)	((obj)->device)
#define cl_query_program(obj)	((obj)->program)
#define cl_query_queue(obj)	((obj)->queue)
#define cl_query_context(obj)	((obj)->context)

// prototypes
#ifdef __cplusplus
extern "C" {
#endif

CL_UTIL_API cl_int cl_Init              (cl_device_type    , cl_device_id *, cl_program *,
					 cl_command_queue *, cl_context   *, const char *, const char *);
CL_UTIL_API void   cl_Fin               (cl_device_id     *, cl_program   *,
					 cl_command_queue *, cl_context   *);
CL_UTIL_API cl_int cl_SetThreadLocalSize(cl_device_id, int, size_t *, size_t *);
CL_UTIL_API void   cl_CheckStatus       (const char *, cl_int);

#ifdef __cplusplus
}
#endif
#undef CL_UTIL_API
#endif
