/*
 * gsm3d.c: Gray-Scott Model Reaction Diffusion System (GS-RDS)
 * (c)2012-2016,2019 Seiji Nishimura
 * $Id: gsm3d.c,v 1.1.1.3 2021/07/17 00:00:00 seiji Exp seiji $
 */

#include <math.h>
#include <time.h>
#include <wtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <libpixmap/window.h>

#ifdef USE_FLOAT
typedef float  real_t;
#else
typedef double real_t;
#endif

#define WIDTH	(0x01<<9)
#define HEIGHT	(0x01<<9)
#define DEPTH	(0x01<<6)

#define STRIDE0	((size_t) (WIDTH ))
#define STRIDE1	((size_t) (HEIGHT))

/* window events */
#define CONTINUE	(0x00)
#define QUIT		(0x01<<0)
#define SAVE_IMAGE	(0x01<<1)
#define TOGGLE_DISPLAY	(0x01<<2)

/* display graphics on window every TI steps */
#define TI	(0x01<<4)
#define DT	(1.0/((real_t)TI))

/* for initial state     */
#define PP	(16*RR*RR*RR)
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
#define ROUND(x)	((int) round(x))
#define CLAMP(x)	(((x)>0xff)?0xff:((x)<0x00)?0x00:(x))
#define MALLOC(n,t)	((t *) malloc((n)*sizeof(t)))

/* 3D array */
#define U(i,j,k)	u[(i)+((j)+(k)*STRIDE1)*STRIDE0]
#define V(i,j,k)	v[(i)+((j)+(k)*STRIDE1)*STRIDE0]
#define P(i,j,k)	p[(i)+((j)+(k)*STRIDE1)*STRIDE0]
#define Q(i,j,k)	q[(i)+((j)+(k)*STRIDE1)*STRIDE0]

/* prototypes ----------------------------------------------------------*/
void init_status  (real_t   *, real_t *);
void update_status(real_t   *, real_t *, real_t *, real_t *);
void draw_image   (window_t *, real_t *, real_t *);
void draw_perf    (window_t *, double  , double);
void save_image   (window_t *, double  );
int  get_input    (window_t *);

/*======================================================================*/
int main(int argc, char **argv)
{
    window_t  w;
    window_t *window = &w;
    real_t *u, *v, *p, *q;

    if ((u = MALLOC(WIDTH*HEIGHT*DEPTH, real_t)) == NULL ||
	(v = MALLOC(WIDTH*HEIGHT*DEPTH, real_t)) == NULL ||
	(p = MALLOC(WIDTH*HEIGHT*DEPTH, real_t)) == NULL ||
	(q = MALLOC(WIDTH*HEIGHT*DEPTH, real_t)) == NULL) {
	perror("malloc");
	return EXIT_FAILURE;
    }

    init_status(u, v);

    window_open(window, WIDTH, HEIGHT, "Gray-Scott Model");

    double t = 0.0;
    bool show_perf = true;

    while (1) {			/* time steps */
	int    action;
	double ts, te;

	ts = wtime();
	draw_image (window, u, v);
	update_status(u, v, p, q);
#ifdef DELAY
	usleep((unsigned int) (DELAY * 1E6));
#endif
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

    free(q); free(p);
    free(v); free(u);

    return EXIT_SUCCESS;
}

/*----------------------------------------------------------------------*/
void init_status(real_t *u, real_t *v)
{				/* setup initial status.                */
    int i , j , k ;
    int ii, jj, kk;

    SRAND((int) time(NULL));	/* initialize RNG seed. */

#pragma omp parallel for private(i,j,k) collapse(3)
    for (k = 0; k < DEPTH; k++) {
	for (j = 0; j < HEIGHT; j++) {
	    for (i = 0; i < WIDTH; i++) {	/* stable status */
		U(i,j,k) = US;
		V(i,j,k) = VS;
	    }
	}
    }

    for (k = 0; k < DEPTH; k++) {
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
		    /* put a 3D ball at (i,j,k) */
		    for (kk = -RR; kk <= RR; kk++) {
			int z = (k + kk + DEPTH) % DEPTH;
			for (jj = -RR; jj <= RR; jj++) {
			    int y = (j + jj + HEIGHT) % HEIGHT;
			    for (ii = -RR; ii <= RR; ii++) {
				int x = (i + ii + WIDTH) % WIDTH;
				if (ii * ii + jj * jj + kk * kk < RR * RR) {
				    U(x,y,z) = uu;
				    V(x,y,z) = vv;
				}
			    }
			}
		    }
		}
	    }
	}
    }

    return;
}

/*----------------------------------------------------------------------*/
void update_status(real_t *u, real_t *v, real_t *p, real_t *q)
{				/* update status.                       */
    int t;
    int ii, jj, kk;

#pragma omp parallel private(t,ii,jj,kk)
    for (t = 0; t < TI; t++) {
#pragma omp for collapse(3)
	for (kk = 0; kk < DEPTH; kk++) {
	    for (jj = 0; jj < HEIGHT; jj++) {
		for (ii = 0; ii < WIDTH; ii++) {	/* p=du/dt, q=dv/dt */
		    int kp = kk + 1, km = kk - 1,
			jp = jj + 1, jm = jj - 1,
			ip = ii + 1, im = ii - 1;
		    if (kp > DEPTH  - 1) kp = 0;
		    if (jp > HEIGHT - 1) jp = 0;
		    if (ip > WIDTH  - 1) ip = 0;
		    if (km < 0) km = DEPTH  - 1;
		    if (jm < 0) jm = HEIGHT - 1;
		    if (im < 0) im = WIDTH  - 1;
		    real_t uu   = U(ii,jj,kk),
			   vv   = V(ii,jj,kk);
		    real_t uvv  = uu * vv * vv;
		    P(ii,jj,kk) = DU * (U(ip,jj,kk) + U(im,jj,kk) +
					U(ii,jp,kk) + U(ii,jm,kk) +
					U(ii,jj,kp) + U(ii,jj,km) - 6.0 * uu) -
					uvv +  F * (1.0 - uu);
		    Q(ii,jj,kk) = DV * (V(ip,jj,kk) + V(im,jj,kk) +
					V(ii,jp,kk) + V(ii,jm,kk) +
					V(ii,jj,kp) + V(ii,jj,km) - 6.0 * vv) +
					uvv - (F + K)   * vv ;
		}
	    }
	}
#pragma omp for collapse(3)
	for (kk = 0; kk < DEPTH; kk++) {
	    for (jj = 0; jj < HEIGHT; jj++) {
		for (ii = 0; ii < WIDTH; ii++) {	/* Euler method */
		    U(ii,jj,kk) += DT * P(ii,jj,kk);
		    V(ii,jj,kk) += DT * Q(ii,jj,kk);
		}
	    }
	}
    }

    return;
}

/*----------------------------------------------------------------------*/
void draw_image(window_t *window, real_t *u, real_t *v)
{				/* update graphics on the window.       */
    int i, j;
    int k = DEPTH / 2;

#pragma omp parallel for private(i,j) collapse(2)
    for (j = 0; j < HEIGHT; j++) {
	for (i = 0; i < WIDTH; i++) {
	    real_t  uu    = U(i,j,k),
		    vv    = V(i,j,k);
	    pixel_t pixel = pixel_set_rgb(CLAMP(ROUND(R_U * uu + R_V * vv + R_C)),
					  CLAMP(ROUND(G_U * uu + G_V * vv + G_C)),
					  CLAMP(ROUND(B_U * uu + B_V * vv + B_C)));
	    window_put_pixel(window, pixel, i, j);
	}
    }

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
