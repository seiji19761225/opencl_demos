/*
 * cl_util.c: utilities for OpenCL parallel processing
 * (c)2017 Seiji Nishimura
 * $Id: cl_util.c,v 1.1.1.1 2020/07/29 00:00:00 seiji Exp seiji $
 */

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stddef.h>
#include "cl_util_internal.h"

#define N	(0x01<<8)

// prototype of internal procedures
static char *_cl_load_kernel_src(const char *, size_t *);
static void  _cl_set_local_size (int, int, size_t *, size_t *);

//----------------------------------------------------------------------
cl_int cl_Init
	(cl_device_type dev_type, cl_device_id *device , cl_program *program,
	 cl_command_queue *queue, cl_context   *context, const char *kernel , const char *build_options)
{				// initalize OpenCL
    cl_platform_id platform[N] = { NULL };
    cl_device_id   dev_id  [N] = { NULL };
    cl_int status, num_devices, num_platforms;
    size_t kernel_size;
    char  *kernel_src ;

    *device  = NULL;
    *program = NULL;
    *queue   = NULL;
    *context = NULL;

    status = clGetPlatformIDs(N, platform, &num_platforms);
    cl_CheckStatus("clGetPlatformIDs", status);

    for (int i = 0; i < num_platforms; i++) {	// search all available platforms.
	status = clGetDeviceIDs(platform[i], dev_type, N, dev_id, &num_devices);
	if (num_devices > 0 &&
	    status == CL_SUCCESS) {
	    *device = dev_id[0];	// use 1st device.
	    break;
	}
    }

    if (*device == NULL) {	// could not find any appropriate device.
	assert(dev_type != CL_DEVICE_TYPE_DEFAULT &&
	       dev_type != CL_DEVICE_TYPE_ALL);
	return CL_DEVICE_NOT_FOUND;
    }

    *context = clCreateContext     (NULL, 1, device, NULL, NULL, &status);
    cl_CheckStatus("clCreateContext"          , status);
    *queue   = clCreateCommandQueue(*context, *device, 0, &status);
    cl_CheckStatus("clCreateCommandQueue"     , status);

    if ((kernel_src = _cl_load_kernel_src(kernel, &kernel_size)) == NULL)
	exit(EXIT_FAILURE);

    // JIT compile OpenCL kernel
    *program = clCreateProgramWithSource(*context, 1, (const char **) &kernel_src, &kernel_size, &status);
    cl_CheckStatus("clCreateProgramWithSource", status);
    status   = clBuildProgram           (*program, 1, device, build_options, NULL, NULL);
    cl_CheckStatus("clBuildProgram"           , status);

    free(kernel_src);

    return CL_SUCCESS;
}

//----------------------------------------------------------------------
void cl_Fin
	(cl_device_id     *device, cl_program *program,
	 cl_command_queue *queue , cl_context *context)
{				// finalize OpenCL
    clReleaseProgram     (*program);
    clReleaseCommandQueue(*queue  );
    clReleaseContext     (*context);

    return;
}

//----------------------------------------------------------------------
cl_int cl_SetThreadLocalSize(cl_device_id device, int ndim, size_t *global_size, size_t *local_size)
{				// determine thread local size for OpenCL
    size_t max_locals = 0;
    cl_int status;

    status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
				sizeof(max_locals), &max_locals, NULL);
    if (status == CL_SUCCESS)
	_cl_set_local_size(ndim, max_locals, global_size, local_size);

    return status;
}

//----------------------------------------------------------------------
void cl_CheckStatus(const char *message, cl_int status)
{				// check returned status, and exit if error happend
    if (status == CL_SUCCESS)
	return;

    fprintf(stderr, "%s: ", message);

    switch (status) {
    case -1 : fprintf(stderr, "Device not found\n"                ); break;
    case -2 : fprintf(stderr, "Device not available\n"            ); break;
    case -3 : fprintf(stderr, "Compiler not available\n"          ); break;
    case -4 : fprintf(stderr, "Memory object allocation failure\n"); break;
    case -5 : fprintf(stderr, "Out of resources\n"                ); break;
    case -6 : fprintf(stderr, "Out of host memory\n"              ); break;
    case -7 : fprintf(stderr, "Profiling info not available\n"    ); break;
    case -8 : fprintf(stderr, "Memory copy overlap\n"             ); break;
    case -9 : fprintf(stderr, "Image format mismatch\n"           ); break;
    case -10: fprintf(stderr, "Image format not supported\n"      ); break;
    case -11: fprintf(stderr, "Build program failure\n"           ); break;
    case -12: fprintf(stderr, "Map failure\n"                     ); break;
    case -30: fprintf(stderr, "Invalid value\n"                   ); break;
    case -31: fprintf(stderr, "Invaid device type\n"              ); break;
    case -32: fprintf(stderr, "Invalid platform\n"                ); break;
    case -33: fprintf(stderr, "Invalid device\n"                  ); break;
    case -34: fprintf(stderr, "Invalid context\n"                 ); break;
    case -35: fprintf(stderr, "Invalid queue properties\n"        ); break;
    case -36: fprintf(stderr, "Invalid command queue\n"           ); break;
    case -37: fprintf(stderr, "Invalid host pointer\n"            ); break;
    case -38: fprintf(stderr, "Invalid memory object\n"           ); break;
    case -39: fprintf(stderr, "Invalid image format descriptor\n" ); break;
    case -40: fprintf(stderr, "Invalid image size\n"              ); break;
    case -41: fprintf(stderr, "Invalid sampler\n"                 ); break;
    case -42: fprintf(stderr, "Invalid binary\n"                  ); break;
    case -43: fprintf(stderr, "Invalid build options\n"           ); break;
    case -44: fprintf(stderr, "Invalid program\n"                 ); break;
    case -45: fprintf(stderr, "Invalid program executable\n"      ); break;
    case -46: fprintf(stderr, "Invalid kernel name\n"             ); break;
    case -47: fprintf(stderr, "Invalid kernel defintion\n"        ); break;
    case -48: fprintf(stderr, "Invalid kernel\n"                  ); break;
    case -49: fprintf(stderr, "Invalid argument index\n"          ); break;
    case -50: fprintf(stderr, "Invalid argument value\n"          ); break;
    case -51: fprintf(stderr, "Invalid argument size\n"           ); break;
    case -52: fprintf(stderr, "Invalid kernel arguments\n"        ); break;
    case -53: fprintf(stderr, "Invalid work dimension\n"          ); break;
    case -54: fprintf(stderr, "Invalid work group size\n"         ); break;
    case -55: fprintf(stderr, "Invalid work item size\n"          ); break;
    case -56: fprintf(stderr, "Invalid global offset\n"           ); break;
    case -57: fprintf(stderr, "Invalid event wait list\n"         ); break;
    case -58: fprintf(stderr, "Invalid event\n"                   ); break;
    case -59: fprintf(stderr, "Invalid operation\n"               ); break;
    case -60: fprintf(stderr, "Invalid GL object\n"               ); break;
    case -61: fprintf(stderr, "Invalid buffer size\n"             ); break;
    case -62: fprintf(stderr, "Invalid mip level\n"               ); break;
    case -63: fprintf(stderr, "Invalid global work size\n"        ); break;
    default : fprintf(stderr, "Unknown error %d\n", status        ); break;
    }

    exit(EXIT_FAILURE);

    return;
}

//----------------------------------------------------------------------
static char *_cl_load_kernel_src(const char *kernel, size_t *kernel_size)
{				// load OpenCL kernel source code
    FILE *fp         = NULL;
    char *kernel_src = NULL;

    // read OpenCL kernel source code
    if ((fp = fopen(kernel, "r")) == NULL) {
	perror(kernel);
	return NULL;
    }

    fseek(fp, 0, SEEK_END);
    *kernel_size = ftell(fp);
    rewind(fp);

    if ((kernel_src   = (char *) malloc(*kernel_size * sizeof(char))) == NULL ||
	*kernel_size != fread(kernel_src, sizeof(char), *kernel_size, fp)) {
	perror(kernel);
	free  (kernel_src);
	fclose(fp);
	return NULL;
    }

    fclose(fp);

    return kernel_src;
}

//----------------------------------------------------------------------
static void _cl_set_local_size(int ndim, int max_locals, size_t *global_size, size_t *local_size)
#if 1
{				// determine thread local size for OpenCL
    if (ndim-- > 0) {
	int x = (int) pow(max_locals, 1.0 / (ndim+1));
	while (global_size[ndim] % x)
	    x--;
	local_size[ndim] = x;
	_cl_set_local_size(ndim, max_locals / x, global_size, local_size);
    }

    return;
}
#else				//......................................
{				// determine thread local size for OpenCL
    if (ndim > 0) {
	int x = (int) pow(max_locals, 1.0 / ndim);
	while (*global_size % x)
	    x--;
	*local_size = x;
	_cl_set_local_size(--ndim, max_locals / x, ++global_size, ++local_size);
    }

    return;
}
#endif
