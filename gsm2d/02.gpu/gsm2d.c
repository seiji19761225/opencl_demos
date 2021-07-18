/*
 * gsm2d.c: Gray-Scott Model Reaction Diffusion System (GS-RDS)
 * (c)2012-2016,2019 Seiji Nishimura
 * $Id: gsm2d.c,v 1.1.1.2 2021/07/17 00:00:00 seiji Exp seiji $
 */

#include <time.h>
#include <wtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <cl_util.h>
#include <libpixmap/window.h>

#ifdef USE_FLOAT
typedef float  real_t;
#else
typedef double real_t;
#endif

#define WIDTH	(0x01<<9)
#define HEIGHT	(0x01<<9)

#ifdef USE_LOCAL_MEMORY
// local size
#define LW	(0x01<<4)
#define LH	(0x01<<4)
#endif

#define STRIDE0	((size_t) (WIDTH))
#ifdef USE_LOCAL_MEMORY
#define STRIDE1	((size_t) (LW+2 ))
#endif

/* window events */
#define CONTINUE	(0x00)
#define QUIT		(0x01<<0)
#define SAVE_IMAGE	(0x01<<1)
#define TOGGLE_DISPLAY	(0x01<<2)

/* display graphics on window every TI steps */
#define TI	(0x01<<9)
#define DT	(1.0/((real_t)TI))

/* for initial state     */
#define PP	(8*RR*RR)
/* stable state (Us, Vs) */
#define US	1.0
#define VS	0.0

/* coloring parameters */
#define R_U	-256.0
#define R_V	 256.0
#define R_C	 256.0
#define G_U	   0.0
#define G_V	1024.0
#define G_C	   0.0
#define B_U	-512.0
#define B_V	   0.0
#define B_C	 512.0

/* macro functions -----------------------------------------------------*/
#define SRAND(s)	srand(s)
#define IRAND()		rand()
#define DRAND()		(((real_t) rand())/((real_t) RAND_MAX+1.0))
#define MALLOC(n,t)	((t *) malloc((n)*sizeof(t)))

/* 2D array */
#define U(i,j)	u[(i)+(j)*STRIDE0]
#define V(i,j)	v[(i)+(j)*STRIDE0]

/* prototypes ----------------------------------------------------------*/
void init_status  (cl_obj_t *, cl_mem    , cl_mem);
void update_status(cl_obj_t *, cl_kernel , cl_kernel, cl_mem, cl_mem, cl_mem, cl_mem);
void draw_image   (window_t *, cl_obj_t *, cl_kernel, cl_mem, cl_mem, cl_mem, cl_mem);
void draw_perf    (window_t *, double    , double);
void save_image   (window_t *, double);
int  get_input    (window_t *);

/*======================================================================*/
int main(int argc, char **argv)
{
    window_t  w;
    window_t *window = &w;
    cl_obj_t  o;
    cl_obj_t *obj    = &o;
    cl_program program;
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel_fdm  ,
	      kernel_euler,
	      kernel_update_image;
    cl_mem dev_u, dev_v, dev_p, dev_q, dev_m, dev_image;
    real_t m[] = {	// 3x3 coloring matrix
	R_U, R_V, R_C,
	G_U, G_V, G_C,
	B_U, B_V, B_C
    };
    char *option =
#ifdef USE_FLOAT
	"-cl-single-precision-constant "
	"-DUSE_FLOAT "
#endif
#ifdef USE_LOCAL_MEMORY
	"-DUSE_LOCAL_MEMORY "
#endif
#if SIZEOF_PIXEL_T == 3
	"-DSIZEOF_PIXEL_T=3"
#else
	"-DSIZEOF_PIXEL_T=4"
#endif
    ;

    // initialize OpenCL
    cl_init(obj, NULL, OPENCL_DEVICE, 0, "./kernel.cl", option);
    program = cl_query_program(obj);
    context = cl_query_context(obj);
    queue   = cl_query_queue  (obj);

    // load GPU kernels
    kernel_fdm          = clCreateKernel(program, "fdm"         , NULL);
    kernel_euler        = clCreateKernel(program, "euler"       , NULL);
    kernel_update_image = clCreateKernel(program, "update_image", NULL);

    // memory allocation on GPU
    dev_u     = clCreateBuffer(context, CL_MEM_READ_WRITE,
				STRIDE0*HEIGHT*sizeof(real_t), NULL, NULL);
    dev_v     = clCreateBuffer(context, CL_MEM_READ_WRITE,
				STRIDE0*HEIGHT*sizeof(real_t), NULL, NULL);
    dev_p     = clCreateBuffer(context, CL_MEM_READ_WRITE,
				STRIDE0*HEIGHT*sizeof(real_t), NULL, NULL);
    dev_q     = clCreateBuffer(context, CL_MEM_READ_WRITE,
				STRIDE0*HEIGHT*sizeof(real_t), NULL, NULL);
    dev_m     = clCreateBuffer(context, CL_MEM_READ_ONLY,
					     9*sizeof(real_t), NULL, NULL);
    dev_image = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				WIDTH  *HEIGHT*SIZEOF_PIXEL_T, NULL, NULL);

    // set up initial state
    init_status(obj, dev_u, dev_v);

    // CPU->GPU memory copy
    clEnqueueWriteBuffer(queue, dev_m, CL_FALSE, 0,
					    9*sizeof(real_t), m, 0, NULL, NULL);

    window_open(window, WIDTH, HEIGHT, "Gray-Scott Model");

    double t = 0.0;
    bool show_perf = true;

    while (1) {			/* time steps */
	int action;
	double ts, te;

	ts = wtime();
	draw_image(window, obj, kernel_update_image, dev_m, dev_u, dev_v, dev_image   );
	update_status     (obj, kernel_fdm, kernel_euler,   dev_u, dev_v, dev_p, dev_q);
#ifdef DELAY
	usleep((unsigned int) (DELAY * 1E6));
#endif
	clFlush (queue);
	clFinish(queue);
	te = wtime();

	if (action = get_input(window)) {
	    if (action & TOGGLE_DISPLAY)
		show_perf = !show_perf;
	    if (action & SAVE_IMAGE)
		save_image(window, t);
	    if (action & QUIT)
		break;
	}

	if (show_perf)
	    draw_perf(window, t, TI / (te - ts));

	window_update_image(window);

	t += DT * TI;
    }

    window_close(window);

    // memory deallocation on GPU
    clReleaseMemObject(dev_image);
    clReleaseMemObject(dev_m);
    clReleaseMemObject(dev_q);
    clReleaseMemObject(dev_p);
    clReleaseMemObject(dev_v);
    clReleaseMemObject(dev_u);

    // unload GPU kernels
    clReleaseKernel(kernel_update_image);
    clReleaseKernel(kernel_euler);
    clReleaseKernel(kernel_fdm);

    // finalize OpenCL
    cl_fin(obj);

    return EXIT_SUCCESS;
}

/*----------------------------------------------------------------------*/
void init_status(cl_obj_t *obj, cl_mem dev_u, cl_mem dev_v)
{				/* setup initial status.                */
    cl_command_queue queue = cl_query_queue(obj);
    int i, j, ii, jj;
    real_t *u, *v;

    SRAND((int) time(NULL));	/* initialize RNG seed. */

    if ((u = MALLOC(STRIDE0*HEIGHT, real_t)) == NULL ||
	(v = MALLOC(STRIDE0*HEIGHT, real_t)) == NULL) {
	perror("malloc");
	exit(EXIT_FAILURE);
    }

#pragma omp parallel for private(i,j) collapse(2)
    for (j = 0; j < HEIGHT; j++) {
	for (i = 0; i < WIDTH; i++) {	/* stable status */
	    U(i,j) = US;
	    V(i,j) = VS;
	}
    }

    for (j = 0; j < HEIGHT; j++) {
	for (i = 0; i < WIDTH; i++) {
	    if (IRAND() % PP == 0) {	/* random seed */
		real_t uu, vv;
#if defined(U0) && defined(V0)
		uu = U0;
		vv = V0;
#else
		uu = DRAND();
		vv = DRAND();
#endif
		/* put a 2D ball at (i,j) */
		for (jj = -RR; jj <= RR; jj++) {
		    int y = (j + jj + HEIGHT) % HEIGHT;
		    for (ii = -RR; ii <= RR; ii++) {
			int x = (i + ii + WIDTH) % WIDTH;
			if (ii * ii + jj * jj < RR * RR) {
			    U(x,y) = uu;
			    V(x,y) = vv;
			}
		    }
		}
	    }
	}
    }

    // CPU->GPU memory copy
    clEnqueueWriteBuffer(queue, dev_u, CL_TRUE, 0,
			 STRIDE0*HEIGHT*sizeof(real_t), u, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, dev_v, CL_TRUE, 0,
			 STRIDE0*HEIGHT*sizeof(real_t), v, 0, NULL, NULL);

    free(v); free(u);

    return;
}

/*----------------------------------------------------------------------*/
void update_status
	(cl_obj_t *obj, cl_kernel kernel_fdm, cl_kernel kernel_euler,
		cl_mem dev_u, cl_mem dev_v, cl_mem dev_p, cl_mem dev_q)
{				/* update status.                       */
    int i;
    int width = WIDTH, height = HEIGHT;
    real_t dt = DT, du = DU, dv = DV, f = F, k = K;
#ifdef USE_LOCAL_MEMORY
    size_t global_size[] = { ROUND_UP(width,LW), ROUND_UP(height,LH) };
    size_t  local_size[] = { LW, LH };
#else
    size_t global_size[] = { ROUND_UP(width,32), ROUND_UP(height,32) };
    size_t *local_size   = NULL;
#endif
    cl_command_queue queue = cl_query_queue(obj);

    // set up kernel args for kernel_fdm
    i = 0;
    clSetKernelArg(kernel_fdm, i++, sizeof(int   ), &width );
    clSetKernelArg(kernel_fdm, i++, sizeof(int   ), &height);
    clSetKernelArg(kernel_fdm, i++, sizeof(real_t), &du    );
    clSetKernelArg(kernel_fdm, i++, sizeof(real_t), &dv    );
    clSetKernelArg(kernel_fdm, i++, sizeof(real_t), &f     );
    clSetKernelArg(kernel_fdm, i++, sizeof(real_t), &k     );
    clSetKernelArg(kernel_fdm, i++, sizeof(cl_mem), &dev_u );
    clSetKernelArg(kernel_fdm, i++, sizeof(cl_mem), &dev_v );
    clSetKernelArg(kernel_fdm, i++, sizeof(cl_mem), &dev_p );
    clSetKernelArg(kernel_fdm, i++, sizeof(cl_mem), &dev_q );
#ifdef USE_LOCAL_MEMORY
    // dynamically allocated local memory
    clSetKernelArg(kernel_fdm, i++, 2*STRIDE1*(LH+2)*
				    sizeof(real_t),    NULL);
#endif

    // set up kernel args for kernel_euler
    i = 0;
    clSetKernelArg(kernel_euler, i++, sizeof(int   ), &width );
    clSetKernelArg(kernel_euler, i++, sizeof(int   ), &height);
    clSetKernelArg(kernel_euler, i++, sizeof(real_t), &dt    );
    clSetKernelArg(kernel_euler, i++, sizeof(cl_mem), &dev_u );
    clSetKernelArg(kernel_euler, i++, sizeof(cl_mem), &dev_v );
    clSetKernelArg(kernel_euler, i++, sizeof(cl_mem), &dev_p );
    clSetKernelArg(kernel_euler, i++, sizeof(cl_mem), &dev_q );

    for (i = 0; i < TI; i++) {	// time integration
	clEnqueueNDRangeKernel(queue, kernel_fdm,
				2, NULL, global_size, local_size, 0, NULL, NULL);
	clEnqueueNDRangeKernel(queue, kernel_euler,
				2, NULL, global_size, local_size, 0, NULL, NULL);
    }

    return;
}

/*----------------------------------------------------------------------*/
void draw_image
	(window_t *window, cl_obj_t *obj, cl_kernel kernel_update_image,
			cl_mem dev_m, cl_mem dev_u, cl_mem dev_v, cl_mem dev_image)
{				/* update graphics on the window.       */
    int i;
    int width = WIDTH, height = HEIGHT;
    size_t global_size[] = { ROUND_UP(width,32), ROUND_UP(height,32) };
    cl_command_queue queue = cl_query_queue(obj);

    // set up kernel args for kernel_update_image
    i = 0;
    clSetKernelArg(kernel_update_image, i++, sizeof(int   ), &width    );
    clSetKernelArg(kernel_update_image, i++, sizeof(int   ), &height   );
    clSetKernelArg(kernel_update_image, i++, sizeof(cl_mem), &dev_m    );
    clSetKernelArg(kernel_update_image, i++, sizeof(cl_mem), &dev_u    );
    clSetKernelArg(kernel_update_image, i++, sizeof(cl_mem), &dev_v    );
    clSetKernelArg(kernel_update_image, i++, sizeof(cl_mem), &dev_image);

    // invoke kernel_update_image
    clEnqueueNDRangeKernel(queue, kernel_update_image,
				2, NULL, global_size, NULL, 0, NULL, NULL);

    // GPU->CPU memory copy
    clEnqueueReadBuffer(queue, dev_image, CL_FALSE, 0,
	  width*height*SIZEOF_PIXEL_T, window->pixmap.data, 0, NULL, NULL);

    return;
}

/*----------------------------------------------------------------------*/
void draw_perf(window_t *window, double t, double fps)
{				/* display performance data.            */
    int i, j;
    int x = 10,
	y = 10;
    char str[256];
    pixel_t black = pixel_set_rgb(0x00, 0x00, 0x00),
	    white = pixel_set_rgb(0xff, 0xff, 0xff);

    snprintf(str, sizeof(str), "T=%10.0f, %10.3f[fps]", t, fps);

    for (j = y-1; j <= y+1; j++) {
	for (i = x-1; i <= x+1; i++) {
	    if (i != x ||
		j != y) {
		window_draw_string(window, black, i, j, str);
	    }
	}
    }

    window_draw_string(window, white, x, y, str);

    return;
}

/*----------------------------------------------------------------------*/
void save_image(window_t *window, double t)
{				/* save current image.                  */
    char fname[256];

    snprintf(fname, sizeof(fname), "output_%010.0f.ppm", t);

    window_write_pnmfile(window, fname);

    return;
}

/*----------------------------------------------------------------------*/
int get_input(window_t *window)
{				/* window event loop                    */
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
	    case('s'):
		action |= SAVE_IMAGE;
		break;
	    default:
		break;
	    }
	}
    } while (device != WD_Null);	// "device==WD_Null" means no event in the queue.

    return action;
}
