/*
 * kernel.cl for gsm3d
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

#define STRIDE0	(width )
#define STRIDE1	(height)
#ifdef USE_LOCAL_MEMORY
#define STRIDE2	(lw+2  )
#define STRIDE3	(lh+2  )
#endif

// 3D array
#define U(i,j,k)	u[(i)+((j)+(k)*STRIDE1)*STRIDE0]
#define V(i,j,k)	v[(i)+((j)+(k)*STRIDE1)*STRIDE0]
#define P(i,j,k)	p[(i)+((j)+(k)*STRIDE1)*STRIDE0]
#define Q(i,j,k)	q[(i)+((j)+(k)*STRIDE1)*STRIDE0]
#ifdef USE_LOCAL_MEMORY
#define A(x,y,z)	a[(x)+((y)+(z)*STRIDE3)*STRIDE2]
#define B(x,y,z)	b[(x)+((y)+(z)*STRIDE3)*STRIDE2]
#endif

//----------------------------------------------------------------------
#ifdef USE_LOCAL_MEMORY
__kernel void fdm
	(int width, int height, int depth,
	 real_t du, real_t  dv,
	 real_t  f, real_t   k,
	 __global real_t *u, __global real_t *v,
	 __global real_t *p, __global real_t *q, __local real_t *local_memory)
{
    int ii = get_global_id(0),
	jj = get_global_id(1),
	kk = get_global_id(2);

    if (ii > width  - 1) ii -= width ;
    if (jj > height - 1) jj -= height;
    if (kk > depth  - 1) kk -= depth ;

    int kp = kk + 1, km = kk - 1,
	jp = jj + 1, jm = jj - 1,
    	ip = ii + 1, im = ii - 1;

    if (kp > depth  - 1) kp = 0;
    if (jp > height - 1) jp = 0;
    if (ip > width  - 1) ip = 0;
    if (km < 0) km = depth  - 1;
    if (jm < 0) jm = height - 1;
    if (im < 0) im = width  - 1;

    // local size
    int lw = get_local_size(0),	// local width
	lh = get_local_size(1),	// local height
	ld = get_local_size(2);	// local depth

    __local real_t *a = &local_memory[0],
		   *b = &local_memory[STRIDE2*STRIDE3*(ld+2)];

    int xx = get_local_id(0) + 1,
	yy = get_local_id(1) + 1,
	zz = get_local_id(2) + 1;

    int zp = zz + 1, zm = zz - 1,
	yp = yy + 1, ym = yy - 1,
    	xp = xx + 1, xm = xx - 1;

    real_t uu   = U(ii,jj,kk),
	   vv   = V(ii,jj,kk);

    A(xx,yy,zz) = uu;
    B(xx,yy,zz) = vv;

    if (xx == 1) {
	A(xm,yy,zz) = U(im,jj,kk);
	B(xm,yy,zz) = V(im,jj,kk);
    }

    if (xx == lw) {
	A(xp,yy,zz) = U(ip,jj,kk);
	B(xp,yy,zz) = V(ip,jj,kk);
    }

    if (yy == 1) {
	A(xx,ym,zz) = U(ii,jm,kk);
	B(xx,ym,zz) = V(ii,jm,kk);
    }

    if (yy == lh) {
	A(xx,yp,zz) = U(ii,jp,kk);
	B(xx,yp,zz) = V(ii,jp,kk);
    }

    if (zz == 1) {
	A(xx,yy,zm) = U(ii,jj,km);
	B(xx,yy,zm) = V(ii,jj,km);
    }

    if (zz == ld) {
	A(xx,yy,zp) = U(ii,jj,kp);
	B(xx,yy,zp) = V(ii,jj,kp);
    }

    // memory fence
    barrier(CLK_LOCAL_MEM_FENCE);

    real_t uvv  = uu * vv * vv;

    P(ii,jj,kk) = du * (A(xp,yy,zz) + A(xm,yy,zz) +
			A(xx,yp,zz) + A(xx,ym,zz) +
			A(xx,yy,zp) + A(xx,yy,zm) - 6.0 * uu) -
		    uvv +  f * (1.0 - uu);

    Q(ii,jj,kk) = dv * (B(xp,yy,zz) + B(xm,yy,zz) +
			B(xx,yp,zz) + B(xx,ym,zz) +
			B(xx,yy,zp) + B(xx,yy,zm) - 6.0 * vv) +
		    uvv - (f + k)   * vv ;

    return;
}
#else				//......................................
__kernel void fdm
	(int width, int height, int depth,
	 real_t du, real_t  dv,
	 real_t  f, real_t   k,
	 __global real_t *u, __global real_t *v,
	 __global real_t *p, __global real_t *q)
{
    int ii = get_global_id(0),
	jj = get_global_id(1),
	kk = get_global_id(2);

    if (ii > width  - 1 ||
	jj > height - 1 ||
	kk > depth  - 1)
	return;

    int kp = kk + 1, km = kk - 1,
	jp = jj + 1, jm = jj - 1,
    	ip = ii + 1, im = ii - 1;

    if (kp > depth  - 1) kp = 0;
    if (jp > height - 1) jp = 0;
    if (ip > width  - 1) ip = 0;
    if (km < 0) km = depth  - 1;
    if (jm < 0) jm = height - 1;
    if (im < 0) im = width  - 1;

    real_t uu   = U(ii,jj,kk),
	   vv   = V(ii,jj,kk);

    real_t uvv  = uu * vv * vv;

    P(ii,jj,kk) = du * (U(ip,jj,kk) + U(im,jj,kk) +
			U(ii,jp,kk) + U(ii,jm,kk) +
			U(ii,jj,kp) + U(ii,jj,km) - 6.0 * uu) -
		    uvv +  f * (1.0 - uu);

    Q(ii,jj,kk) = dv * (V(ip,jj,kk) + V(im,jj,kk) +
			V(ii,jp,kk) + V(ii,jm,kk) +
			V(ii,jj,kp) + V(ii,jj,km) - 6.0 * vv) +
		    uvv - (f + k)   * vv ;

    return;
}
#endif

//----------------------------------------------------------------------
__kernel void euler
	(int width, int height, int depth, real_t dt,
	 __global real_t *u, __global real_t *v, __global real_t *p, __global real_t *q)
{
    int i = get_global_id(0),
	j = get_global_id(1),
	k = get_global_id(2);

    if (i > width  - 1 ||
	j > height - 1 ||
	k > depth  - 1)
	return;

    U(i,j,k) += dt * P(i,j,k);
    V(i,j,k) += dt * Q(i,j,k);

    return;
}

//----------------------------------------------------------------------
__kernel void update_image
	(int width, int height, int depth,
	 __constant real_t *m, __global real_t *u, __global real_t *v, __global uchar *image)
#if SIZEOF_PIXEL_T == 4
{
    int i = get_global_id(0),
	j = get_global_id(1),
	k = depth / 2;

    if (i > width  - 1 ||
	j > height - 1)
	return;

    real_t uu = U(i,j,k),
	   vv = V(i,j,k);

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
	j = get_global_id(1),
	k = depth / 2;

    if (i > width  - 1 ||
	j > height - 1)
	return;

    real_t uu = U(i,j,k),
	   vv = V(i,j,k);

    real_t rr = m[0] * uu + m[1] * vv + m[2],
	   gg = m[3] * uu + m[4] * vv + m[5],
	   bb = m[6] * uu + m[7] * vv + m[8];

    uchar3  pixel = convert_uchar3_sat(round((real3_t) (rr, gg, bb)));

    vstore3(pixel, i + j * width, image);

    return;
}
#endif
