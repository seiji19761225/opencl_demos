/*
 * ao.c: aobench is originally written by Syoyo Fujita.
 * (c)2019 Seiji Nishimura
 * $Id: kernel.cl,v 1.1.1.1 2020/07/29 00:00:00 seiji Exp seiji $
 */

#ifdef USE_FLOAT
typedef float   real_t;
typedef float3  real3_t;
typedef float4  real4_t;
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double  real_t;
typedef double3 real3_t;
typedef double4 real4_t;
#endif

typedef unsigned long seed_t;

typedef real3_t vec;

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

#ifdef USE_NATIVE_MATH
#define NATIVE_FUNC(func)	native_ ## func
#define sin	NATIVE_FUNC(sin)
#define cos	NATIVE_FUNC(cos)
#define sqrt	NATIVE_FUNC(sqrt)
#endif

// prototypes
void   init_scene          (Sphere *, Plane *, real_t);
void   render              (__global uchar  *, Sphere *, Plane *, int, int, int, int);
vec    ambient_occlusion   (const Isect     *, Sphere *, Plane *, seed_t *, int);
void   orthoBasis          (vec    *, vec);
void   ray_sphere_intersect(Isect  *, const Ray *, Sphere *);
void   ray_plane_intersect (Isect  *, const Ray *, Plane  *);
real_t drand64_r           (seed_t *);

//----------------------------------------------------------------------
__kernel void aobench
	(__global uchar *image, int width, int height, int nsubsamples, int nao_samples, real_t theta)
{
    Sphere spheres[3];
    Plane  plane;

    init_scene(spheres, &plane, theta);

    render(image, spheres, &plane, width, height, nsubsamples, nao_samples);

    return;
}

//----------------------------------------------------------------------
void init_scene(Sphere *spheres, Plane *plane, real_t theta)
{
    spheres[0].center = (vec) (-2.0, 0.0, -3.5);
    spheres[0].radius =  0.5;

    spheres[1].center = (vec) (-0.5, 0.5 * (1.0 - cos(theta)), -3.0);
    spheres[1].radius =  0.5;

    spheres[2].center = (vec) ( 1.0, 0.0, -2.2);
    spheres[2].radius =  0.5;

    plane->p = (vec) (0.0, -0.5, 0.0);
    plane->n = (vec) (0.0,  1.0, 0.0);

    return;
}

//----------------------------------------------------------------------
void render
	(__global uchar *image, Sphere *spheres, Plane *plane,
	 int width, int height, int nsubsamples, int nao_samples)
{
    int x = get_global_id(0),
	y = get_global_id(1);

    if (x >= width ||
	y >= height)
	return;

    real_t d = 2.0 / min(width, height);

    seed_t seed = ULONG_MAX - (x + y * width);

    vec col = 0.0;

    for (int v = 0; v < nsubsamples; v++) {
	for (int u = 0; u < nsubsamples; u++) {
	    real_t px =  (x + (u / (real_t) nsubsamples) - (width  / 2.0)) * d;
	    real_t py = -(y + (v / (real_t) nsubsamples) - (height / 2.0)) * d;

	    Ray ray;

	    ray.org = 0.0;
	    ray.dir = (vec) (px, py, -1.0);
	    ray.dir = normalize(ray.dir);

	    Isect isect;
	    isect.t   = 1.0e+17;
	    isect.hit = 0;

	    ray_sphere_intersect(&isect, &ray, &spheres[0]);
	    ray_sphere_intersect(&isect, &ray, &spheres[1]);
	    ray_sphere_intersect(&isect, &ray, &spheres[2]);
	    ray_plane_intersect (&isect, &ray,  plane);

	    if (isect.hit)
		col += ambient_occlusion(&isect, spheres, plane, &seed, nao_samples);
	}
    }

    col *= 256.0 / (nsubsamples * nsubsamples);

#if SIZEOF_PIXEL_T == 3
    uchar3  pixel = convert_uchar3_sat(col);
    vstore3(pixel, x + y * width, image);
#else
    uchar4  pixel = convert_uchar4_sat((real4_t) (0.0, col));
    vstore4(pixel, x + y * width, image);
#endif

    return;
}

//----------------------------------------------------------------------
vec ambient_occlusion
	(const Isect *isect, Sphere *spheres, Plane *plane, seed_t *seed, int nao_samples)
{
    int ntheta = nao_samples;
    int nphi   = nao_samples;
    real_t eps = 0.0001;

    vec p = isect->p + eps * isect->n;
    vec basis[3];

    orthoBasis(basis, isect->n);

    real_t occlusion = 0.0;

    for (int j = 0; j < ntheta; j++) {
	for (int i = 0; i < nphi; i++) {
	    real_t theta =         sqrt(drand64_r(seed));
	    real_t phi   = 2.0 * M_PI * drand64_r(seed) ;
	    real_t x     = cos(phi) * theta;
	    real_t y     = sin(phi) * theta;
	    real_t z     = sqrt(1.0 - theta * theta);

	    Ray ray;

	    ray.org = p;
	    ray.dir = x * basis[0] + y * basis[1] + z * basis[2];	// local -> global

	    Isect occIsect;
	    occIsect.t   = 1.0e+17;
	    occIsect.hit = 0;

	    ray_sphere_intersect(&occIsect, &ray, &spheres[0]);
	    ray_sphere_intersect(&occIsect, &ray, &spheres[1]);
	    ray_sphere_intersect(&occIsect, &ray, &spheres[2]);
	    ray_plane_intersect (&occIsect, &ray,  plane);

	    if (occIsect.hit)
		occlusion += 1.0;
	}
    }

    p = (ntheta * nphi - occlusion) / (real_t) (ntheta * nphi);

    return p;
}

//----------------------------------------------------------------------
void orthoBasis(vec *basis, vec n)
{
    basis[1] = 0.0;
    basis[2] = n  ;

    if (n.x < 0.6 && n.x > -0.6) {
	basis[1].x = 1.0;
    } else if (n.y < 0.6 && n.y > -0.6) {
	basis[1].y = 1.0;
    } else if (n.z < 0.6 && n.z > -0.6) {
	basis[1].z = 1.0;
    } else {
	basis[1].x = 1.0;
    }

    basis[0] = cross    (basis[1], basis[2]);
    basis[0] = normalize(basis[0]);

    basis[1] = cross    (basis[2], basis[0]);
    basis[1] = normalize(basis[1]);

    return;
}

//----------------------------------------------------------------------
void ray_sphere_intersect(Isect *isect, const Ray *ray, Sphere *sphere)
{
    vec rs = ray->org - sphere->center;

    real_t B = dot(rs, ray->dir);
    real_t C = dot(rs, rs) - sphere->radius * sphere->radius;
    real_t D = B * B - C;

    if (D > 0.0) {
	real_t t = -B - sqrt(D);
	if (t > 0.0 && t < isect->t) {
	    isect->t   = t;
	    isect->hit = 1;

	    isect->p = ray->org + ray->dir * t;
	    isect->n = isect->p - sphere->center;
	    isect->n = normalize(isect->n);
	}
    }

    return;
}

//----------------------------------------------------------------------
void ray_plane_intersect(Isect *isect, const Ray *ray, Plane *plane)
{
    real_t d = -dot(plane->p, plane->n);
    real_t v =  dot(ray->dir, plane->n);

    if (fabs(v) < 1.0e-17)
	return;

    real_t t = -(dot(ray->org, plane->n) + d) / v;

    if (t > 0.0 && t < isect->t) {
	isect->t   = t;
	isect->hit = 1;

	isect->p = ray->org + ray->dir * t;
	isect->n = plane->n;
    }

    return;
}

//----------------------------------------------------------------------
real_t drand64_r(seed_t *x)
{				// xorshift64 random number generator
    *x = *x ^ (*x << 13);
    *x = *x ^ (*x >>  7);
    *x = *x ^ (*x << 17);

    return (real_t) *x / ULONG_MAX;
}
