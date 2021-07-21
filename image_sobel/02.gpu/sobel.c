/*
 * sobel.c
 * (c)2019 Seiji Nishimura
 * $Id: sobel.c,v 1.1.1.2 2021/07/18 00:00:00 seiji Exp seiji $
 */

#include <wtime.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <cl_util.h>
#include <CL/opencl.h>
#include <libpixmap/pixmap.h>

typedef unsigned char uchar;

#define OUTPUT_FILE	"output.pgm"

#ifdef USE_LOCAL_MEMORY
// local size
#define LW	(0x01<<4)
#define LH	(0x01<<4)
#endif

#define MIN(x,y)	(((x)<(y))?(x):(y))
#define MALLOC(n,t)	((t *) malloc((n)*sizeof(t)))
#define RGB2GRAY(r,g,b)	((int) (0.299*(r)+0.587*(g)+0.114*(b)+0.5))

// 2D array
#define INPUT(i,j)	( input[(i)+(j)*width])
#define OUTPUT(i,j)	(output[(i)+(j)*width])

cl_mem load_image  (cl_obj_t *, const char *, int *, int *);
void   save_image  (cl_obj_t *, const char *, int  , int  , cl_mem);
void   sobel_filter(cl_obj_t *, cl_kernel   , int  , int  , cl_mem, cl_mem);

//======================================================================
int main(int argc, char **argv)
{
    cl_obj_t  o;
    cl_obj_t *obj = &o;
    cl_program program;
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel_sobel;
    cl_mem    dev_i, dev_o;
    int width, height;
    double ts, te, speed;
    char *option =
#ifdef USE_LOCAL_MEMORY
	"-DUSE_LOCAL_MEMORY "
#endif
	"-cl-single-precision-constant";

    // initialize OpenCL
    cl_init(obj, NULL, OPENCL_DEVICE, 0, "./kernel.cl", option);
    program = cl_query_program(obj);
    context = cl_query_context(obj);
    queue   = cl_query_queue  (obj);

    // load GPU kernel
    kernel_sobel = clCreateKernel(program, "SobelFilter", NULL);

    // memory allocation on GPU
    dev_i = load_image(obj, INPUT_FILE, &width, &height);
    dev_o = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				width * height * sizeof(uchar), NULL, NULL);

    ts = wtime();
    sobel_filter(obj, kernel_sobel, width, height, dev_i, dev_o);
    te = wtime();

    speed = 1.E-6 * width * height / (te - ts);

    printf("%dx%d: T=%f[msec.], %f[MP/s]\n", width, height, (te - ts) * 1000.0, speed);

    save_image(obj, OUTPUT_FILE, width, height, dev_o);

    // memory deallocation on GPU
    clReleaseMemObject(dev_o);
    clReleaseMemObject(dev_i);

    // unload GPU kernels
    clReleaseKernel(kernel_sobel);

    // finalize OpenCL
    cl_fin(obj);

    return EXIT_SUCCESS;
}

//----------------------------------------------------------------------
cl_mem load_image(cl_obj_t *obj, const char *fname, int *width, int *height)
{
    int i, j;
    pixmap_t pixmap ;
    uchar *image    ;
    cl_mem dev_image;
    cl_context     context = cl_query_context(obj);
    cl_command_queue queue = cl_query_queue  (obj);

    if (pixmap_load_pnmfile(&pixmap, fname) == EXIT_FAILURE) {
	if (errno) {	/* system error */
	    perror(fname);
	} else {	/* unacceptable file format */
	    fprintf(stderr, "%s: Not a ppm, pgm, or pbm file\n", fname);
	}
	exit(EXIT_FAILURE);
    }

    *width  = pixmap.width ;
    *height = pixmap.height;

    if ((image = MALLOC(*width * *height, uchar)) == NULL) {
	perror(__func__);
	exit(EXIT_FAILURE);
    }

    // convert image from pixmap to graymap
#pragma omp parallel for private(i,j) collapse(2)
    for (j = 0; j < *height; j++) {
	for (i = 0; i < *width; i++) {
	    pixel_t pixel;
	    pixmap_get_pixel(&pixmap, &pixel, i, j);
	    image[i + j * *width] = RGB2GRAY(pixel_get_r(pixel),
					     pixel_get_g(pixel),
					     pixel_get_b(pixel));
	}
    }

    // memory allocation on GPU
    dev_image = clCreateBuffer(context, CL_MEM_READ_ONLY,
				*width * *height * sizeof(uchar), NULL, NULL);

    // CPU->GPU memory copy
    clEnqueueWriteBuffer(queue, dev_image, CL_TRUE, 0,
				*width * *height * sizeof(uchar), image, 0, NULL, NULL);

    pixmap_destroy(&pixmap);
    free(image);

    return dev_image;
}

//----------------------------------------------------------------------
void save_image(cl_obj_t *obj, const char *fname, int width, int height, cl_mem dev_image)
{
    uchar *image;
    FILE  *fp = NULL;
    cl_command_queue queue = cl_query_queue(obj);

    if ((image = MALLOC(width * height, uchar)) == NULL) {
	perror(__func__);
	exit(EXIT_FAILURE);
    }

    // GPU->CPU memory copy
    clEnqueueReadBuffer(queue, dev_image, CL_TRUE, 0,
			width * height * sizeof(uchar), image, 0, NULL, NULL);

    if ((fp = fopen(fname, "wb")) == NULL) {
	perror(fname);
	exit(EXIT_FAILURE);
    }

    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    fwrite(image, sizeof(uchar), width * height, fp);

    fclose(fp);

    free(image);

    return;
}

//----------------------------------------------------------------------
void sobel_filter(cl_obj_t *obj, cl_kernel kernel, int width, int height,
		  cl_mem dev_input, cl_mem dev_output)
{
#ifdef USE_LOCAL_MEMORY
    size_t global_size[] = { ROUND_UP(width, LW), ROUND_UP(height, LH) };
    size_t  local_size[] = { LW, LH };
#else
    size_t global_size[] = { ROUND_UP(width, 32), ROUND_UP(height, 32) };
    size_t *local_size   = NULL;
#endif
    cl_command_queue queue = cl_query_queue(obj);

    // set up kernel args
    int i = 0;
    clSetKernelArg(kernel, i++, sizeof(int   ), &width     );
    clSetKernelArg(kernel, i++, sizeof(int   ), &height    );
    clSetKernelArg(kernel, i++, sizeof(cl_mem), &dev_input );
    clSetKernelArg(kernel, i++, sizeof(cl_mem), &dev_output);
#ifdef USE_LOCAL_MEMORY
    // dynamically allocated local memory
    clSetKernelArg(kernel, i++, (LW+2)*(LH+2)*sizeof(uchar), NULL);
#endif

    // invoke kernel
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

    clFinish(queue);

    return;
}
