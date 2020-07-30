/*
 * ao.c: aobench is originally written by Syoyo Fujita.
 * (c)2019 Seiji Nishimura
 * $Id: ao.c,v 1.1.1.1 2020/07/29 00:00:00 seiji Exp seiji $
 */

#include <omp.h>
#include <stdio.h>
#include <wtime.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <tgmath.h>
#include <cl_util.h>
#include <CL/opencl.h>
#include <libpixmap/window.h>

#ifdef USE_FLOAT
typedef float  real_t;
#else
typedef double real_t;
#endif

typedef struct _vec {
    real_t x;
    real_t y;
    real_t z;
} vec;

typedef struct _Sphere {
    vec center;
    real_t radius;
} Sphere;

typedef struct _Plane {
    vec p;
    vec n;
} Plane;

/* window events */
#define CONTINUE	(0x00)
#define QUIT		(0x01<<0)
#define TOGGLE_DISPLAY	(0x01<<1)

// prototypes
void   init_scene(cl_obj_t *, cl_mem, cl_mem, real_t);
void   render    (window_t *, cl_obj_t *, cl_kernel, cl_mem, cl_mem, cl_mem, int, int, int, int);
void   draw_perf (window_t *, double);
int    get_input (window_t *);

//======================================================================
int main(int argc, char **argv)
{
    window_t  w;
    window_t *window = &w;
    cl_obj_t  o;
    cl_obj_t *obj    = &o;
    cl_program program;
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    cl_mem    dev_spheres, dev_plane, dev_image;
    char *option =
#ifdef USE_FLOAT
	"-cl-single-precision-constant "
	"-DUSE_FLOAT "
#endif
#ifdef USE_NATIVE_MATH
	"-DUSE_NATIVE_MATH "
#endif
#if SIZEOF_PIXMAP_T == 3
	"-DSIZEOF_PIXMAP_T=3";
#else
	"-DSIZEOF_PIXMAP_T=4";
#endif

    // initialize OpenCL
    cl_init(obj, OPENCL_DEVICE, "./kernel.cl", option);
    program = cl_query_program(obj);
    context = cl_query_context(obj);
    queue   = cl_query_queue  (obj);

    // load GPU kernel
    kernel  = clCreateKernel(program, "render", NULL);

    // memory allocation on GPU
    dev_spheres = clCreateBuffer(context, CL_MEM_READ_ONLY ,
				3 * sizeof(Sphere), NULL, NULL);
    dev_plane   = clCreateBuffer(context, CL_MEM_READ_ONLY ,
				    sizeof(Plane ), NULL, NULL);
    dev_image   = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		   WIDTH * HEIGHT * SIZEOF_PIXEL_T, NULL, NULL);

    window_open(window, WIDTH, HEIGHT, "AOBench");

    real_t   theta =  0.0;
    bool show_perf = true;

    while (1) {
	int    action = CONTINUE;
	double ts, te;

	init_scene(obj, dev_spheres, dev_plane, theta);

	ts = wtime();
	render(window, obj, kernel, dev_image, dev_spheres, dev_plane,
				WIDTH, HEIGHT, NSUBSAMPLES, NAO_SAMPLES);
#ifdef DELAY
	usleep((unsigned int) (DELAY * 1E6));
#endif
	clFlush (queue);
	clFinish(queue);
	te = wtime();

	if (action = get_input(window)) {
	    if (action & TOGGLE_DISPLAY)
		show_perf = !show_perf;
	    if (action & QUIT)
		break;
	}

	if (show_perf)
	    draw_perf(window, te - ts);

	window_update_image(window);

	theta += 2.0 * M_PI / NFRAMES;
	if (theta >  2.0 * M_PI)
	    theta -= 2.0 * M_PI;
    }

    window_close(window);

    // memory deallocation on GPU
    clReleaseMemObject(dev_image);
    clReleaseMemObject(dev_plane);
    clReleaseMemObject(dev_spheres);

    // finalize OpenCL
    cl_fin(obj);

    return EXIT_SUCCESS;
}

//----------------------------------------------------------------------
void init_scene(cl_obj_t *obj, cl_mem dev_spheres, cl_mem dev_plane, real_t theta)
{
    cl_command_queue queue = cl_query_queue(obj);
    Plane  plane;
    Sphere spheres[3];

    spheres[0].center.x = -2.0;
    spheres[0].center.y =  0.0;
    spheres[0].center.z = -3.5;
    spheres[0].radius   =  0.5;

    spheres[1].center.x = -0.5;
    spheres[1].center.y =  0.5 * (1.0 - cos(theta));
    spheres[1].center.z = -3.0;
    spheres[1].radius   =  0.5;

    spheres[2].center.x =  1.0;
    spheres[2].center.y =  0.0;
    spheres[2].center.z = -2.2;
    spheres[2].radius   =  0.5;

    plane.p.x =  0.0;
    plane.p.y = -0.5;
    plane.p.z =  0.0;

    plane.n.x =  0.0;
    plane.n.y =  1.0;
    plane.n.z =  0.0;

    // CPU->GPU memory copy
    clEnqueueWriteBuffer(queue, dev_spheres, CL_TRUE, 0,
				3 * sizeof(Sphere), spheres, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, dev_plane  , CL_TRUE, 0,
				    sizeof(Plane ), &plane , 0, NULL, NULL);

    return;
}

//----------------------------------------------------------------------
void render
	(window_t *window, cl_obj_t *obj, cl_kernel kernel, cl_mem dev_image, cl_mem dev_spheres, cl_mem dev_plane,
	 int width, int height, int nsubsamples, int nao_samples)
{
    cl_command_queue queue = cl_query_queue(obj);
    size_t global_size[]   = { ROUND_UP(width,32), ROUND_UP(height,32) };
    size_t *local_size     = NULL;

    // set up kernel args
    int i = 0;
    clSetKernelArg(kernel, i++, sizeof(cl_mem), &dev_image  );
    clSetKernelArg(kernel, i++, sizeof(cl_mem), &dev_spheres);
    clSetKernelArg(kernel, i++, sizeof(cl_mem), &dev_plane  );
    clSetKernelArg(kernel, i++, sizeof(int   ), &width      );
    clSetKernelArg(kernel, i++, sizeof(int   ), &height     );
    clSetKernelArg(kernel, i++, sizeof(int   ), &nsubsamples);
    clSetKernelArg(kernel, i++, sizeof(int   ), &nao_samples);

    clEnqueueNDRangeKernel(queue, kernel,
		2, NULL, global_size, local_size, 0, NULL, NULL);

    // GPU->CPU memory copy
    clEnqueueReadBuffer(queue, dev_image, CL_FALSE, 0,
		width*height*SIZEOF_PIXEL_T, window->pixmap.data, 0, NULL, NULL);

    return;
}

//----------------------------------------------------------------------
void draw_perf(window_t *window, double t)
{				// display performance data
    int i, j;
    int x = 10,
	y = 10;
    char str[256];
    pixel_t black = pixel_set_rgb(0x00, 0x00, 0x00),
	    white = pixel_set_rgb(0xff, 0xff, 0xff);

    snprintf(str, sizeof(str), "T=%8.3f[msec.], %8.3f[fps]", t * 1000.0, 1.0 / t);

    for (j = y-1; j <= y+1; j++) {
	for (i = x-1; i <= x+1; i++) {
	    if (i != x || j != y) {
		window_draw_string(window, black, i, j, str);
	    }
	}
    }

    window_draw_string(window, white, x, y, str);

    return;
}

//----------------------------------------------------------------------
int get_input(window_t *window)
{				// window event loop
    int action = CONTINUE;
    int device, code, x, y;

    do {
	window_get_input(window, false, &device, &code, &x, &y);
	if (device == WD_KeyBoard) {
	    switch(code) {
	    case(WK_ESC):
		action |= QUIT;
		break;
	    case(' '):
		action |= TOGGLE_DISPLAY;
		break;
	    default:
		break;
	    }
	}
    } while (device != WD_Null);	// "device==WD_Null" means no event in the queue.

    return action;
}
