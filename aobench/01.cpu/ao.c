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
#include <libpixmap/window.h>

#ifdef USE_FLOAT
typedef float  real_t;
#else
typedef double real_t;
#endif

typedef uint8_t   uchar;
typedef uint64_t seed_t;

typedef struct _vec {
    real_t x;
    real_t y;
    real_t z;
} vec;

typedef struct _Isect {
    real_t t;
    vec p;
    vec n;
    int hit;
} Isect;

typedef struct _Sphere {
    vec center;
    real_t radius;
} Sphere;

typedef struct _Plane {
    vec p;
    vec n;
} Plane;

typedef struct _Ray {
    vec org;
    vec dir;
} Ray;

/* window events */
#define CONTINUE	(0x00)
#define QUIT		(0x01<<0)
#define TOGGLE_DISPLAY	(0x01<<1)

#define MIN(x,y)	(((x)<(y))?(x):(y))

// prototypes
void   init_scene          (Sphere   *, Plane  *, real_t);
void   render              (window_t *, int, int, Sphere *, Plane *);
vec    ambient_occlusion   (const Isect *, Sphere *, Plane *, seed_t *, int);
void   orthoBasis          (vec      *, vec);
void   ray_sphere_intersect(Isect    *, const Ray *, const Sphere *);
void   ray_plane_intersect (Isect    *, const Ray *, const Plane  *);
real_t vdot                (vec       , vec);
vec    vcross              (vec       , vec);
vec    vnormalize          (vec       );
uchar  clamp               (real_t    );
real_t drand64_r           (seed_t   *);
void   draw_perf           (window_t *, double);
int    get_input           (window_t *);

//======================================================================
int main(int argc, char **argv)
{
    window_t  w;
    window_t *window = &w;

    window_open(window, WIDTH, HEIGHT, "AOBench");

    real_t   theta =  0.0;
    bool show_perf = true;

    while (1) {
	int    action = CONTINUE;
	double ts, te;
	Plane  plane;
	Sphere spheres[3];

	init_scene(spheres, &plane, theta);

	ts = wtime();
#pragma omp parallel
	render(window, NSUBSAMPLES, NAO_SAMPLES, spheres, &plane);
#ifdef DELAY
	usleep((unsigned int) (DELAY * 1E6));
#endif
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

    return EXIT_SUCCESS;
}

//----------------------------------------------------------------------
void init_scene(Sphere *spheres, Plane *plane, real_t theta)
{
    spheres[0].center.x = -2.0;
    spheres[0].center.y =  0.0;
    spheres[0].center.z = -3.5;
    spheres[0].radius   =  0.5;

    spheres[1].center.x = -0.5;
    spheres[1].center.y = -0.5 * cos(theta) + 0.5;
    spheres[1].center.z = -3.0;
    spheres[1].radius   =  0.5;

    spheres[2].center.x =  1.0;
    spheres[2].center.y =  0.0;
    spheres[2].center.z = -2.2;
    spheres[2].radius   =  0.5;

    plane->p.x =  0.0;
    plane->p.y = -0.5;
    plane->p.z =  0.0;

    plane->n.x =  0.0;
    plane->n.y =  1.0;
    plane->n.z =  0.0;

    return;
}

//----------------------------------------------------------------------
void render(window_t *window, int nsubsamples, int nao_samples, Sphere *spheres, Plane *plane)
{
    int x, y;
    int u, v;
    int w, h;

    window_get_size(window, &w, &h);

    real_t d = 2.0 / MIN(w, h);

    seed_t seed = UINT64_MAX - omp_get_thread_num();

#pragma omp for collapse(2) schedule(dynamic) nowait
    for (y = 0; y < h; y++) {
	for (x = 0; x < w; x++) {
	    pixel_t pixel;
	    real_t rr = 0.0,
		   gg = 0.0,
		   bb = 0.0;
	    for (v = 0; v < nsubsamples; v++) {
		for (u = 0; u < nsubsamples; u++) {
		    real_t px =  (x + (u / (real_t) nsubsamples) - (w / 2.0)) * d;
		    real_t py = -(y + (v / (real_t) nsubsamples) - (h / 2.0)) * d;

		    Ray ray;

		    ray.org.x =  0.0;
		    ray.org.y =  0.0;
		    ray.org.z =  0.0;

		    ray.dir.x =   px;
		    ray.dir.y =   py;
		    ray.dir.z = -1.0;
		    ray.dir   = vnormalize(ray.dir);

		    Isect isect;
		    isect.t   = 1.0e+17;
		    isect.hit = 0;

		    ray_sphere_intersect(&isect, &ray, &spheres[0]);
		    ray_sphere_intersect(&isect, &ray, &spheres[1]);
		    ray_sphere_intersect(&isect, &ray, &spheres[2]);
		    ray_plane_intersect (&isect, &ray,  plane);

		    if (isect.hit) {
			vec col = ambient_occlusion(&isect, spheres, plane, &seed, nao_samples);
			rr += col.x;
			gg += col.y;
			bb += col.z;
		    }
		}
	    }
	    pixel = pixel_set_rgb(clamp(rr / (real_t) (nsubsamples * nsubsamples)),
				  clamp(gg / (real_t) (nsubsamples * nsubsamples)),
				  clamp(bb / (real_t) (nsubsamples * nsubsamples)));
	    window_put_pixel(window, pixel, x, y);
	}
    }

    return;
}

//----------------------------------------------------------------------
vec ambient_occlusion(const Isect *isect, Sphere *spheres, Plane *plane, seed_t *seed, int nao_samples)
{
    int i, j;
    int ntheta = nao_samples;
    int nphi   = nao_samples;
    real_t eps = 0.0001;

    vec p;

    p.x = isect->p.x + eps * isect->n.x;
    p.y = isect->p.y + eps * isect->n.y;
    p.z = isect->p.z + eps * isect->n.z;

    vec basis[3];

    orthoBasis(basis, isect->n);

    real_t occlusion = 0.0;

    for (j = 0; j < ntheta; j++) {
	for (i = 0; i < nphi; i++) {
	    real_t theta =         sqrt(drand64_r(seed));
	    real_t phi   = 2.0 * M_PI * drand64_r(seed) ;

	    real_t x = cos(phi) * theta;
	    real_t y = sin(phi) * theta;
	    real_t z = sqrt(1.0 - theta * theta);

	    // local -> global
	    real_t rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
	    real_t ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
	    real_t rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

	    Ray ray;

	    ray.org   = p ;
	    ray.dir.x = rx;
	    ray.dir.y = ry;
	    ray.dir.z = rz;

	    Isect occIsect;
	    occIsect.t = 1.0e+17;
	    occIsect.hit = 0;

	    ray_sphere_intersect(&occIsect, &ray, &spheres[0]);
	    ray_sphere_intersect(&occIsect, &ray, &spheres[1]);
	    ray_sphere_intersect(&occIsect, &ray, &spheres[2]);
	    ray_plane_intersect (&occIsect, &ray,  plane);

	    if (occIsect.hit)
		occlusion += 1.0;
	}
    }

    occlusion = (ntheta * nphi - occlusion) / (real_t) (ntheta * nphi);

    p.x = occlusion;
    p.y = occlusion;
    p.z = occlusion;

    return p;
}

//----------------------------------------------------------------------
void orthoBasis(vec *basis, vec n)
{
    basis[1].x = 0.0;
    basis[1].y = 0.0;
    basis[1].z = 0.0;
    basis[2]   = n  ;

    if (n.x < 0.6 && n.x > -0.6) {
	basis[1].x = 1.0;
    } else if (n.y < 0.6 && n.y > -0.6) {
	basis[1].y = 1.0;
    } else if (n.z < 0.6 && n.z > -0.6) {
	basis[1].z = 1.0;
    } else {
	basis[1].x = 1.0;
    }

    basis[0] = vcross    (basis[1], basis[2]);
    basis[0] = vnormalize(basis[0]);

    basis[1] = vcross    (basis[2], basis[0]);
    basis[1] = vnormalize(basis[1]);

    return;
}

//----------------------------------------------------------------------
real_t vdot(vec v0, vec v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

//----------------------------------------------------------------------
void ray_sphere_intersect(Isect *isect, const Ray *ray, const Sphere *sphere)
{
    vec rs;

    rs.x = ray->org.x - sphere->center.x;
    rs.y = ray->org.y - sphere->center.y;
    rs.z = ray->org.z - sphere->center.z;

    real_t B = vdot(rs, ray->dir);
    real_t C = vdot(rs, rs) - sphere->radius * sphere->radius;
    real_t D = B * B - C;

    if (D > 0.0) {
	real_t t = -B - sqrt(D);
	if (t > 0.0 && t < isect->t) {
	    isect->t   = t;
	    isect->hit = 1;

	    isect->p.x = ray->org.x + ray->dir.x * t;
	    isect->p.y = ray->org.y + ray->dir.y * t;
	    isect->p.z = ray->org.z + ray->dir.z * t;

	    isect->n.x = isect->p.x - sphere->center.x;
	    isect->n.y = isect->p.y - sphere->center.y;
	    isect->n.z = isect->p.z - sphere->center.z;
	    isect->n   = vnormalize(isect->n);
	}
    }

    return;
}

//----------------------------------------------------------------------
void ray_plane_intersect(Isect *isect, const Ray *ray, const Plane *plane)
{
    real_t d = -vdot(plane->p, plane->n);
    real_t v =  vdot(ray->dir, plane->n);

    if (fabs(v) < 1.0e-17)
	return;

    real_t t = -(vdot(ray->org, plane->n) + d) / v;

    if (t > 0.0 && t < isect->t) {
	isect->t   = t;
	isect->hit = 1;

	isect->p.x = ray->org.x + ray->dir.x * t;
	isect->p.y = ray->org.y + ray->dir.y * t;
	isect->p.z = ray->org.z + ray->dir.z * t;

	isect->n   = plane->n;
    }

    return;
}

//----------------------------------------------------------------------
vec vcross(vec v0, vec v1)
{
    vec c;

    c.x = v0.y * v1.z - v0.z * v1.y;
    c.y = v0.z * v1.x - v0.x * v1.z;
    c.z = v0.x * v1.y - v0.y * v1.x;

    return c;
}

//----------------------------------------------------------------------
vec vnormalize(vec c)
{
    real_t length = sqrt(vdot(c, c));

    if (fabs(length) > 1.0e-17) {
	c.x /= length;
	c.y /= length;
	c.z /= length;
    }

    return c;
}

//----------------------------------------------------------------------
uchar clamp(real_t f)
{
    int i = (int) (256.0 * f);

    if (i <   0) i =   0;
    if (i > 255) i = 255;

    return (uchar) i;
}

//----------------------------------------------------------------------
real_t drand64_r(seed_t *x)
{				// xorshift64 random number generator
    *x = *x ^ (*x << 13);
    *x = *x ^ (*x >>  7);
    *x = *x ^ (*x << 17);

    return ((real_t) *x) / UINT64_MAX;
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
