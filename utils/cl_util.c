/*
 * cl_util.c: OpenCL utility
 * (c)2017-2021 Seiji Nishimura
 * $Id: cl_util.c,v 1.1.1.1 2020/07/29 00:00:00 seiji Exp seiji $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cl_util_internal.h"

#define N_TBL	(0x01<<8)

// prototypes
static void   clCheckStatus_    (const char *  , cl_int  );
static cl_int clGetPlatformName_(cl_platform_id, char   *, size_t);
static cl_int clGetDeviceName_  (cl_device_id  , char   *, size_t);
static char  *clLoadKernelSrc_  (const char *  , size_t *);

//----------------------------------------------------------------------
void cl_init
	(cl_obj_t *obj, char *platform_name, cl_device_type device_type, cl_uint device_num,
	 char  *kernel, char *build_options)
{				// initialize OpenCL.
    cl_platform_id platform[N_TBL];
    cl_device_id   dev_id  [N_TBL];
    cl_int  status     ;
    cl_uint num_devices, num_platforms;
    size_t  kernel_size;
    char   *kernel_src ;

    obj->device  = NULL;
    obj->program = NULL;
    obj->queue   = NULL;
    obj->context = NULL;

    status = clGetPlatformIDs(N_TBL, platform, &num_platforms);
    clCheckStatus_("clGetPlatformIDs", status);

    if (platform_name == NULL) {	// no specified specific platform.
	for (int i = 0; i < num_platforms; i++) {
	    status = clGetDeviceIDs(platform[i], device_type, N_TBL, dev_id, &num_devices);
//	    clCheckStatus_("clGetDeviceIDs", status);
	    if (num_devices > 0) {
#ifdef DEBUG
		char pname[256];
		status = clGetPlatformName_(platform[i], pname, sizeof(pname));
		clCheckStatus_("clGetPlatformName_", status);
#endif
		if (num_devices > device_num) {	// found the target device.
#ifdef DEBUG
		    char dname[256];
		    status = clGetDeviceName_(dev_id[device_num], dname, sizeof(dname));
		    clCheckStatus_("clGetDeviceName_", status);
		    printf("OpenCL:\n");
		    printf("\tPlatform=%s\n", pname);
		    printf("\tDevice  =%s\n", dname);
#endif
		    obj->device = dev_id[device_num];
		    break;
		} else {
		    device_num -= num_devices;
		}
	    }
	}
    } else {				// specified specific platform.
	char pname[256];
	for (int i = 0; i < num_platforms; i++) {
	    status = clGetPlatformName_(platform[i], pname, sizeof(pname));
	    clCheckStatus_("clGetPlatformName_", status);
	    if (strcmp(platform_name, pname) == 0) {	// found the target platform.
		status = clGetDeviceIDs(platform[i], device_type, N_TBL, dev_id, &num_devices);
//		clCheckStatus_("clGetDeviceIDs", status);
		if (num_devices > 0) {
		    if (num_devices > device_num) {	// found the target device.
#ifdef DEBUG
			char dname[256];
			status = clGetDeviceName_(dev_id[device_num], dname, sizeof(dname));
			clCheckStatus_("clGetDeviceName_", status);
			printf("OpenCL:\n");
			printf("\tPlatform=%s\n", pname);
			printf("\tDevice  =%s\n", dname);
#endif
			obj->device = dev_id[device_num];
			break;
		    } else {
			device_num -= num_devices;
		    }
		}
	    }
	}
    }

    if (obj->device == NULL)	// could not find any appropriate device.
	clCheckStatus_(__func__, CL_DEVICE_NOT_FOUND);

    obj->context = clCreateContext(NULL, 1, &obj->device, NULL, NULL, &status);
    clCheckStatus_("clCreateContext", status);
    obj->queue   = clCreateCommandQueue(obj->context, obj->device, 0, &status);
    clCheckStatus_("clCreateCommandQueue", status);

    if ((kernel_src = clLoadKernelSrc_(kernel, &kernel_size)) == NULL)
	exit(EXIT_FAILURE);

    // JIT compiler OpenCL kernel
    obj->program = clCreateProgramWithSource
			(obj->context, 1, (const char **) &kernel_src, &kernel_size, &status);
    clCheckStatus_("clCreateProgramWithSource", status);
    status       = clBuildProgram(obj->program, 1, &obj->device, build_options, NULL, NULL);
    clCheckStatus_("clBuildProgram", status);

    free(kernel_src);

    return;
}

//----------------------------------------------------------------------
void cl_fin
	(cl_obj_t *obj)
{				// finalize OpenCL.
    clReleaseProgram     (obj->program);
    clReleaseCommandQueue(obj->queue  );
    clReleaseContext     (obj->context);

    return;
}

//----------------------------------------------------------------------
static
void clCheckStatus_
	(const char *message, cl_int status)
{				// check status value, and exit if error happened.
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
static
cl_int clGetPlatformName_
	(cl_platform_id platform, char *platform_name, size_t n)
{				// get name of the specified platform.
    return clGetPlatformInfo(platform, CL_PLATFORM_NAME, n, platform_name, NULL);
}

//----------------------------------------------------------------------
static
cl_int clGetDeviceName_
	(cl_device_id device, char *device_name, size_t n)
{				// get name of the specified device.
    return clGetDeviceInfo(device, CL_DEVICE_NAME, n, device_name, NULL);
}

//----------------------------------------------------------------------
static
char *clLoadKernelSrc_
	(const char *kernel, size_t *kernel_size)
{				// load OpenCL kernel source code.
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
