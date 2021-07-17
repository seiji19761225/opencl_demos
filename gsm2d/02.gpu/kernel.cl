/*
 * kernel.cl for gsm2d
 * (c)2019 Seiji Nishimura
 * $Id: kernel.cl,v 1.1.1.2 2021/07/17 00:00:00 seiji Exp seiji $
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

#define STRIDE0	((size_t) (width))
#ifdef USE_LOCAL_MEMORY
#define STRIDE1	((size_t) (lw+2 ))
#endif

// 2D array
#define U(i,j)	u[(i)+(j)*STRIDE0]
#define V(i,j)	v[(i)+(j)*STRIDE0]
#define P(i,j)	p[(i)+(j)*STRIDE0]
#define Q(i,j)	q[(i)+(j)*STRIDE0]
#ifdef USE_LOCAL_MEMORY
#define A(x,y)	a[(x)+(y)*STRIDE1]
#define B(x,y)	b[(x)+(y)*STRIDE1]
#endif

//----------------------------------------------------------------------
#ifdef USE_LOCAL_MEMORY
__kernel void fdm
	(int width, int height,
	 real_t du, real_t dv, real_t f, real_t k,
	 __global real_t *u, __global real_t *v,
	 __global real_t *p, __global real_t *q, __local real_t *local_memory)
{
    int ii = get_global_id(0),
	jj = get_global_id(1);

    if (ii > width  - 1) ii -= width ;
    if (jj > height - 1) jj -= height;

    int ip = ii + 1, im = ii - 1,
	jp = jj + 1, jm = jj - 1;

    if (ip > width  - 1) ip = 0;
    if (jp > height - 1) jp = 0;
    if (im < 0) im = width  - 1;
    if (jm < 0) jm = height - 1;

    // local size
    int lw = get_local_size(0),	// local width
	lh = get_local_size(1);	// local height

    __local real_t *a = &local_memory[0],
		   *b = &local_memory[STRIDE1*(lh+2)];

    int xx = get_local_id(0) + 1,
	yy = get_local_id(1) + 1;

    int xp = xx + 1, xm = xx - 1,
	yp = yy + 1, ym = yy - 1;

    real_t uu = U(ii,jj),
	   vv = V(ii,jj);

    A(xx,yy) = uu;
    B(xx,yy) = vv;

    if (xx == 1) {
	A(xm,yy) = U(im,jj);
	B(xm,yy) = V(im,jj);
    }

    if (xx == lw) {
	A(xp,yy) = U(ip,jj);
	B(xp,yy) = V(ip,jj);
    }

    if (yy == 1) {
	A(xx,ym) = U(ii,jm);
	B(xx,ym) = V(ii,jm);
    }

    if (yy == lh) {
	A(xx,yp) = U(ii,jp);
	B(xx,yp) = V(ii,jp);
    }

    // memory fence
    barrier(CLK_LOCAL_MEM_FENCE);

    real_t uvv  = uu * vv * vv;

    P(ii,jj) = du * (A(xp,yy) + A(xm,yy) +
		     A(xx,yp) + A(xx,ym) - 4.0 * uu) -
			       uvv +  f * (1.0 - uu);

    Q(ii,jj) = dv * (B(xp,yy) + B(xm,yy) +
		     B(xx,yp) + B(xx,ym) - 4.0 * vv) +
			       uvv - (f + k)   * vv ;

    return;
}
#else				//......................................
__kernel void fdm
	(int width, int height,
	 real_t du, real_t dv, real_t f, real_t k,
	 __global real_t *u, __global real_t *v,
	 __global real_t *p, __global real_t *q)
{
    int ii = get_global_id(0),
	jj = get_global_id(1);

    if (ii > width  - 1 ||
	jj > height - 1)
	return;

    int ip = ii + 1, im = ii - 1,
	jp = jj + 1, jm = jj - 1;

    if (ip > width  - 1) ip = 0;
    if (jp > height - 1) jp = 0;
    if (im < 0) im = width  - 1;
    if (jm < 0) jm = height - 1;

    real_t uu   = U(ii,jj),
	   vv   = V(ii,jj);

    real_t uvv  = uu * vv * vv;

    P(ii,jj) = du * (U(ip,jj) + U(im,jj) +
		     U(ii,jp) + U(ii,jm) - 4.0 * uu) -
			       uvv +  f * (1.0 - uu);

    Q(ii,jj) = dv * (V(ip,jj) + V(im,jj) +
		     V(ii,jp) + V(ii,jm) - 4.0 * vv) +
			       uvv - (f + k)   * vv ;

    return;
}
#endif

//----------------------------------------------------------------------
__kernel void euler
	(int width, int height, real_t dt,
	 __global real_t *u, __global real_t *v, __global real_t *p, __global real_t *q)
{
    int i = get_global_id(0),
	j = get_global_id(1);

    if (i > width  - 1 ||
	j > height - 1)
	return;

    U(i,j) += dt * P(i,j);
    V(i,j) += dt * Q(i,j);

    return;
}

//----------------------------------------------------------------------
__kernel void update_image
	(int width, int height,
	 __constant real_t *m, __global real_t *u, __global real_t *v, __global uchar *image)
#if SIZEOF_PIXEL_T == 4
{
    int i = get_global_id(0),
	j = get_global_id(1);

    if (i > width  - 1 ||
	j > height - 1)
	return;

    real_t uu = U(i,j),
	   vv = V(i,j);

    real_t rr = m[0] * uu + m[1] * vv + m[2],
	   gg = m[3] * uu + m[4] * vv + m[5],
	   bb = m[6] * uu + m[7] * vv + m[8];

    uchar4  pixel = convert_uchar4_sat(round((real4_t) (0.0, rr, gg, bb)));

    vstore4(pixel, i + j * width, image);

    return;
}
#else				//......................................
{
    int i = get_global_id(0),
	j = get_global_id(1);

    if (i > width  - 1 ||
	j > height - 1)
	return;

    real_t uu = U(i,j),
	   vv = V(i,j);

    real_t rr = m[0] * uu + m[1] * vv + m[2],
	   gg = m[3] * uu + m[4] * vv + m[5],
	   bb = m[6] * uu + m[7] * vv + m[8];

    uchar3  pixel = convert_uchar3_sat(round((real3_t) (rr, gg, bb)));

    vstore3(pixel, i + j * width, image);

    return;
}
#endif
