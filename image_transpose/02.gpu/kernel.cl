/*
 * kernel for Transpose
 * (c)2019 Seiji Nishimura
 * $Id: kernel.cl,v 1.1.1.1 2020/07/30 00:00:00 seiji Exp seiji $
 */

#define STRIDE_I	(width)
#define STRIDE_O	(height)
#ifdef USE_LOCAL_MEMORY
#define STRIDE_L	(lh+1)
#endif

// 2D array
#define INPUT(i,j)	(       input[(i)+(j)*STRIDE_I])
#define OUTPUT(i,j)	(      output[(i)+(j)*STRIDE_O])
#ifdef USE_LOCAL_MEMORY
#define LOCAL(i,j)	(local_memory[(i)+(j)*STRIDE_L])
#endif

//----------------------------------------------------------------------
#ifdef USE_LOCAL_MEMORY
__kernel void Transpose
	(int width, int height,
	 __global uchar *input, __global uchar *output, __local uchar *local_memory)
{
    int i  = get_global_id (0),
	j  = get_global_id (1);

    if (i > width  - 1 ||
	j > height - 1)
	return;

    int lw = get_local_size(0),
	lh = get_local_size(1);

    int x  = get_local_id  (0),
	y  = get_local_id  (1);

    LOCAL(y,x) = INPUT(i,j);

    // memory fence
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef USE_REORDERING
    int k = x + y * lw;

    y = k % lh;
    x = k / lh;

    j = y + get_group_id(1) * lh;
    i = x + get_group_id(0) * lw;
#endif

    OUTPUT(j,i) = LOCAL(y,x);

    return;
}
#else				//......................................
__kernel void Transpose
	(int width, int height,
	 __global uchar *input, __global uchar *output)
{
    int i = get_global_id(0),
	j = get_global_id(1);

    if (i > width  - 1 ||
	j > height - 1)
	return;

    OUTPUT(j,i) = INPUT(i,j);

    return;
}
#endif
