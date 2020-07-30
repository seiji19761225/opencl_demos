/*
 * kernel for Sobel
 * (c)2019 Seiji Nishimura
 * $Id: kernel.cl,v 1.1.1.1 2020/07/30 00:00:00 seiji Exp seiji $
 */

#define STRIDE_G	(width)
#ifdef USE_LOCAL_MEMORY
#define STRIDE_L	(lw+2)
#endif

// 2D array
#define INPUT(i,j)	( input[(i)+(j)*STRIDE_G])
#define OUTPUT(i,j)	(output[(i)+(j)*STRIDE_G])
#ifdef USE_LOCAL_MEMORY
#define LOCAL(i,j)	(local_memory[(i)+(j)*STRIDE_L])
#endif

//----------------------------------------------------------------------
#ifdef USE_LOCAL_MEMORY
__kernel void SobelFilter(int width, int height,
			  __global uchar *input, __global uchar *output, __local uchar *local_memory)
{
    bool out_of_boundary = false;
    int ii = get_global_id(0),
	jj = get_global_id(1);

    if (ii > width  - 1) {
	ii = width  - 1;
	out_of_boundary = true;
    }

    if (jj > height - 1) {
	jj = height - 1;
	out_of_boundary = true;
    }

    int im = ii - 1, ip = ii + 1,
	jm = jj - 1, jp = jj + 1;

    ip = min(ip, width  - 1);
    jp = min(jp, height - 1);
    im = max(im, 0);
    jm = max(jm, 0);

    // local size
    int lw = get_local_size(0),
	lh = get_local_size(1);

    int xx = get_local_id(0) + 1,
	yy = get_local_id(1) + 1;

    int yp = yy + 1, ym = yy - 1,
    	xp = xx + 1, xm = xx - 1;

    LOCAL(xx,yy) = INPUT(ii,jj);

    // boundary elements
    if (xx ==  1) LOCAL(xm,yy) = INPUT(im,jj);
    if (xx == lw) LOCAL(xp,yy) = INPUT(ip,jj);
    if (yy ==  1) LOCAL(xx,ym) = INPUT(ii,jm);
    if (yy == lh) LOCAL(xx,yp) = INPUT(ii,jp);

    // corner elements
    if (xx ==  1 &&
	yy ==  1) LOCAL(xm,ym) = INPUT(im,jm);
    if (xx ==  1 &&
	yy == lh) LOCAL(xm,yp) = INPUT(im,jp);
    if (xx == lw &&
	yy ==  1) LOCAL(xp,ym) = INPUT(ip,jm);
    if (xx == lw &&
	yy == lh) LOCAL(xp,yp) = INPUT(ip,jp);

    // memory fence
    barrier(CLK_LOCAL_MEM_FENCE);

    if (out_of_boundary)
	return;

    int sobel, sobel_h, sobel_v;

    sobel_h =     LOCAL(xm,ym)                    -     LOCAL(xp,ym) +
	      2 * LOCAL(xm,yy)                    - 2 * LOCAL(xp,yy) +
		  LOCAL(xm,yp)                    -     LOCAL(xp,yp);
    sobel_v =     LOCAL(xm,ym) + 2 * LOCAL(xx,ym) +     LOCAL(xp,ym) -
		  LOCAL(xm,yp) - 2 * LOCAL(xx,yp) -     LOCAL(xp,yp);
//  sobel   = hypot(sobel_h, sobel_v);
    sobel   = abs(sobel_h) + abs(sobel_v);

    OUTPUT(ii,jj) = convert_uchar_sat(sobel);

    return;
}
#else				//......................................
__kernel void SobelFilter(int width, int height, __global uchar *input, __global uchar *output)
{
    int ii = get_global_id(0),
	jj = get_global_id(1);

    if (ii > width  - 1 ||
	jj > height - 1)
	return;

    int im = ii - 1, ip = ii + 1,
	jm = jj - 1, jp = jj + 1;

    ip = min(ip, width  - 1);
    jp = min(jp, height - 1);
    im = max(im, 0);
    jm = max(jm, 0);

    int sobel, sobel_h, sobel_v;

    sobel_h =     INPUT(im,jm)                    -     INPUT(ip,jm) +
	      2 * INPUT(im,jj)                    - 2 * INPUT(ip,jj) +
		  INPUT(im,jp)                    -     INPUT(ip,jp);
    sobel_v =     INPUT(im,jm) + 2 * INPUT(ii,jm) +     INPUT(ip,jm) -
		  INPUT(im,jp) - 2 * INPUT(ii,jp) -     INPUT(ip,jp);
//  sobel   = hypot(sobel_h, sobel_v);
    sobel   = abs(sobel_h) + abs(sobel_v);

    OUTPUT(ii,jj) = convert_uchar_sat(sobel);

    return;
}
#endif
