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

#ifdef USE_NATIVE_MATH
#define NATIVE_FUNC(func)	native_ ## func
#define sin	NATIVE_FUNC(sin)
#define cos	NATIVE_FUNC(cos)
#define sqrt	NATIVE_FUNC(sqrt)
#endif

// prototypes
vec    ambient_occlusion   (const Isect *, __constant Sphere *, __constant Plane *, seed_t *, int);
void   orthoBasis          (vec    *, vec);
void   ray_sphere_intersect(Isect  *, const Ray *, __constant Sphere *);
void   ray_plane_intersect (Isect  *, const Ray *, __constant Plane  *);
real_t vdot                (vec     , vec);
vec    vcross              (vec     , vec);
vec    vnormalize          (vec     );
real_t drand64_r           (seed_t *);

//----------------------------------------------------------------------
__kernel void render
	(__global uchar *image, __constant Sphere *spheres, __constant Plane *plane,
	 int width, int height, int nsubsamples, int nao_samples)
{
    int x = get_global_id(0),
	y = get_global_id(1);

    if (x >= width ||
	y >= height)
	return;

    real_t d = 2.0 / min(width, height);

    seed_t seed = ULONG_MAX - (x + y * width);

    real_t rr = 0.0,
	   gg = 0.0,
	   bb = 0.0;

    for (int v = 0; v < nsubsamples; v++) {
	for (int u = 0; u < nsubsamples; u++) {
	    real_t px =  (x + (u / (real_t) nsubsamples) - (width  / 2.0)) * d;
	    real_t py = -(y + (v / (real_t) nsubsamples) - (height / 2.0)) * d;

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

#if SIZEOF_PIXEL_T == 3
    real3_t rgb   = (real3_t)      (rr, gg, bb) * 256.0 / (nsubsamples * nsubsamples);
    uchar3  pixel = convert_uchar3_sat(rgb);
    vstore3(pixel, x + y * width, image);
#else
    real4_t rgb   = (real4_t) (0.0, rr, gg, bb) * 256.0 / (nsubsamples * nsubsamples);
    uchar4  pixel = convert_uchar4_sat(rgb);
    vstore4(pixel, x + y * width, image);
#endif

    return;
}

//----------------------------------------------------------------------
vec ambient_occlusion
	(const Isect *isect, __constant Sphere *spheres, __constant Plane *plane, seed_t *seed, int nao_samples)
{
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

    for (int j = 0; j < ntheta; j++) {
	for (int i = 0; i < nphi; i++) {
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
void ray_sphere_intersect(Isect *isect, const Ray *ray, __constant Sphere *sphere)
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
void ray_plane_intersect(Isect *isect, const Ray *ray, __constant Plane *plane)
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
real_t drand64_r(seed_t *x)
{				// xorshift64 random number generator
    *x = *x ^ (*x << 13);
    *x = *x ^ (*x >>  7);
    *x = *x ^ (*x << 17);

    return (real_t) *x / ULONG_MAX;
}
