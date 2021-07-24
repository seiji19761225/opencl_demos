/*
 * cl_util.c: OpenCL utility
 * (c)2017-2021 Seiji Nishimura
 * $Id: cl_util.c,v 1.1.1.5 2021/07/24 00:00:00 seiji Exp seiji $
 */

#include "cl_util_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef CL_UTIL_USE_STAT
#include <fcntl.h>
#include <sys/stat.h>
#endif

typedef enum { BINARY, SOURCE } kernel_type_t;

#define N_TBL		(0x01<<8)

#define size(tbl)	(sizeof(tbl)/sizeof(tbl[0]))

// prototypes
static char         *clLoadKernelSrc_   (const char *, size_t *);
static kernel_type_t clDetectKernelType_(const char *);
static void          clCheckStatus_     (const char *, cl_int  );

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

    status = clGetPlatformIDs(size(platform), platform, &num_platforms);
    cl_check_status(status);

    if (platform_name == NULL) {	// no specified specific platform.
	for (int i = 0; i < num_platforms; i++) {
	    status = clGetDeviceIDs(platform[i], device_type, size(dev_id), dev_id, &num_devices);
	    if (status != CL_DEVICE_NOT_FOUND)
		cl_check_status(status);
	    if (num_devices > 0) {
#ifdef CL_UTIL_DEBUG
		char pname[256];
		status = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, sizeof(pname), pname, NULL);
		cl_check_status(status);
#endif
		if (num_devices > device_num) {	// found the target device.
#ifdef CL_UTIL_DEBUG
		    char dname[256];
		    status = clGetDeviceInfo(dev_id[device_num], CL_DEVICE_NAME,
							sizeof(dname), dname, NULL);
		    cl_check_status(status);
		    printf("OpenCL\n");
		    printf("  Platform: %s\n", pname);
		    printf("  Device  : %s\n", dname);
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
	    status = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, sizeof(pname), pname, NULL);
	    cl_check_status(status);
	    if (strcasecmp(platform_name, pname) == 0) {	// found the target platform.
		status = clGetDeviceIDs(platform[i], device_type, size(dev_id), dev_id, &num_devices);
		if (status != CL_DEVICE_NOT_FOUND)
		    cl_check_status(status);
		if (num_devices > 0) {
		    if (num_devices > device_num) {	// found the target device.
#ifdef CL_UTIL_DEBUG
			char dname[256];
			status = clGetDeviceInfo(dev_id[device_num], CL_DEVICE_NAME,
								sizeof(dname), dname, NULL);
			cl_check_status(status);
			printf("OpenCL\n");
			printf("  Platform: %s\n", pname);
			printf("  Device  : %s\n", dname);
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
	cl_check_status(CL_DEVICE_NOT_FOUND);

    obj->context = clCreateContext(NULL, 1, &obj->device, NULL, NULL, &status);
    cl_check_status(status);
    obj->queue   = clCreateCommandQueue(obj->context, obj->device, 0, &status);
    cl_check_status(status);

    if ((kernel_src = clLoadKernelSrc_(kernel, &kernel_size)) == NULL)
	exit(EXIT_FAILURE);

    // JIT compile OpenCL kernel
    if (clDetectKernelType_(kernel) == SOURCE) {
	obj->program = clCreateProgramWithSource
			(obj->context, 1, (const char **) &kernel_src, &kernel_size, &status);
	cl_check_status(status);
    } else {	// clKernelType_(kernel) == BINARY
	obj->program = clCreateProgramWithBinary
			(obj->context, 1, &obj->device,
			   &kernel_size, (const unsigned char **) &kernel_src, NULL, &status);
	cl_check_status(status);
    }
    status = clBuildProgram(obj->program, 1, &obj->device, build_options, NULL, NULL);
    cl_check_status(status);

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
void cl_check_status_
	(const char *fname, const int line, cl_int status)
#ifdef NDEBUG
{				// debug assertion
    clCheckStatus_(NULL, status);

    return;
}
#else				//......................................
{				// debug assertion
    char buf[256];

    snprintf(buf, sizeof(buf), "Line #%d in %s", line, fname);

    clCheckStatus_((const char *) buf, status);

    return;
}
#endif

//----------------------------------------------------------------------
static
char *clLoadKernelSrc_
	(const char *kernel, size_t *kernel_size)
#ifdef CL_UTIL_USE_STAT
{				// load OpenCL kernel source code.
    struct stat st;
    FILE *fp         = NULL;
    char *kernel_src = NULL;

    if (stat(kernel, &st) != 0 || (fp = fopen(kernel, "r")) == NULL) {
	perror(kernel);
	return NULL;
    }

    *kernel_size = st.st_size;

    // read OpenCL kernel source code
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
#else				//......................................
{				// load OpenCL kernel source code.
    FILE *fp         = NULL;
    char *kernel_src = NULL;

    if ((fp = fopen(kernel, "r")) == NULL) {
	perror(kernel);
	return NULL;
    }

    fseek(fp, 0, SEEK_END);
    *kernel_size = ftell(fp);
    rewind(fp);

    // read OpenCL kernel source code
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
#endif

//----------------------------------------------------------------------
static
kernel_type_t clDetectKernelType_(const char *fname)
{				// detect file type of OpenCL kernel.
    const struct {
	char *suffix;
	kernel_type_t type;
    } type_table[] = {
	{ ".bin", BINARY },
	{ ".ir" , BINARY },
	{ ".lib", BINARY },
	{ ".out", BINARY },
	{ ".cl" , SOURCE },
	{ ".clc", SOURCE },
	{ ".src", SOURCE },
	{ ".txt", SOURCE }
    };

    kernel_type_t kernel_type = SOURCE;

    size_t fname_length = strlen(fname);

    for (int i = 0; i < size(type_table); i++) {
	size_t suffix_length = strlen(type_table[i].suffix);
	if (fname_length > suffix_length &&
	    strcasecmp(&fname[fname_length - suffix_length], type_table[i].suffix) == 0) {
	    kernel_type = type_table[i].type;
	    break;
	}
    }

    return kernel_type;
}

//----------------------------------------------------------------------
static
void clCheckStatus_
	(const char *message, cl_int status)
#ifdef NDEBUG
{				// check status value, and exit if error happened.
    if (status != CL_SUCCESS)
	exit(status);

    return;
}
#else				//......................................
{				// check status value, and exit if error happened.
    if (status == CL_SUCCESS)
	return;

    fprintf(stderr, "%s: ", message);

    switch (status) {
    case -1 : fprintf(stderr, "Device not found.\n"                ); break;
    case -2 : fprintf(stderr, "Device not available.\n"            ); break;
    case -3 : fprintf(stderr, "Compiler not available.\n"          ); break;
    case -4 : fprintf(stderr, "Memory object allocation failure.\n"); break;
    case -5 : fprintf(stderr, "Out of resources.\n"                ); break;
    case -6 : fprintf(stderr, "Out of host memory.\n"              ); break;
    case -7 : fprintf(stderr, "Profiling info not available.\n"    ); break;
    case -8 : fprintf(stderr, "Memory copy overlap.\n"             ); break;
    case -9 : fprintf(stderr, "Image format mismatch.\n"           ); break;
    case -10: fprintf(stderr, "Image format not supported.\n"      ); break;
    case -11: fprintf(stderr, "Build program failure.\n"           ); break;
    case -12: fprintf(stderr, "Map failure.\n"                     ); break;
    case -30: fprintf(stderr, "Invalid value.\n"                   ); break;
    case -31: fprintf(stderr, "Invaid device type.\n"              ); break;
    case -32: fprintf(stderr, "Invalid platform.\n"                ); break;
    case -33: fprintf(stderr, "Invalid device.\n"                  ); break;
    case -34: fprintf(stderr, "Invalid context.\n"                 ); break;
    case -35: fprintf(stderr, "Invalid queue properties.\n"        ); break;
    case -36: fprintf(stderr, "Invalid command queue.\n"           ); break;
    case -37: fprintf(stderr, "Invalid host pointer.\n"            ); break;
    case -38: fprintf(stderr, "Invalid memory object.\n"           ); break;
    case -39: fprintf(stderr, "Invalid image format descriptor.\n" ); break;
    case -40: fprintf(stderr, "Invalid image size.\n"              ); break;
    case -41: fprintf(stderr, "Invalid sampler.\n"                 ); break;
    case -42: fprintf(stderr, "Invalid binary.\n"                  ); break;
    case -43: fprintf(stderr, "Invalid build options.\n"           ); break;
    case -44: fprintf(stderr, "Invalid program.\n"                 ); break;
    case -45: fprintf(stderr, "Invalid program executable.\n"      ); break;
    case -46: fprintf(stderr, "Invalid kernel name.\n"             ); break;
    case -47: fprintf(stderr, "Invalid kernel defintion.\n"        ); break;
    case -48: fprintf(stderr, "Invalid kernel.\n"                  ); break;
    case -49: fprintf(stderr, "Invalid argument index.\n"          ); break;
    case -50: fprintf(stderr, "Invalid argument value.\n"          ); break;
    case -51: fprintf(stderr, "Invalid argument size.\n"           ); break;
    case -52: fprintf(stderr, "Invalid kernel arguments.\n"        ); break;
    case -53: fprintf(stderr, "Invalid work dimension.\n"          ); break;
    case -54: fprintf(stderr, "Invalid work group size.\n"         ); break;
    case -55: fprintf(stderr, "Invalid work item size.\n"          ); break;
    case -56: fprintf(stderr, "Invalid global offset.\n"           ); break;
    case -57: fprintf(stderr, "Invalid event wait list.\n"         ); break;
    case -58: fprintf(stderr, "Invalid event.\n"                   ); break;
    case -59: fprintf(stderr, "Invalid operation.\n"               ); break;
    case -60: fprintf(stderr, "Invalid GL object.\n"               ); break;
    case -61: fprintf(stderr, "Invalid buffer size.\n"             ); break;
    case -62: fprintf(stderr, "Invalid mip level.\n"               ); break;
    case -63: fprintf(stderr, "Invalid global work size.\n"        ); break;
    default : fprintf(stderr, "Unknown error %d.\n", status        ); break;
    }

    exit(EXIT_FAILURE);

    return;
}
#endif
