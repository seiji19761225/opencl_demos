/*
 * sobel.c
 * (c)2019 Seiji Nishimura
 * $Id: sobel.c,v 1.1.1.1 2020/07/30 00:00:00 seiji Exp seiji $
 */

#include <wtime.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <libpixmap/pixmap.h>

typedef unsigned char uchar;

#define OUTPUT_FILE	"output.pgm"

#define MIN(x,y)	(((x)<(y))?(x):(y))
#define MALLOC(n,t)	((t *) malloc((n)*sizeof(t)))
#define RGB2GRAY(r,g,b)	((int) (0.299*(r)+0.587*(g)+0.114*(b)+0.5))

// 2D array
#define INPUT(i,j)	( input[(i)+(j)*width])
#define OUTPUT(i,j)	(output[(i)+(j)*width])

uchar *load_image  (const char *, int *, int *);
void   save_image  (const char *, int  , int  , uchar *);
void   sobel_filter(int, int, uchar *, uchar *);

//======================================================================
int main(int argc, char **argv)
{
    int width, height;
    uchar *input  = NULL,
	  *output = NULL;
    double ts, te, speed;

    if ((input  = load_image(INPUT_FILE, &width, &height)) == NULL ||
	(output = MALLOC(width * height, uchar)) == NULL) {
	perror("malloc");
	return EXIT_FAILURE;
    }

    ts = wtime();
    sobel_filter(width, height, input, output);
    te = wtime();

    speed = 1.E-6 * width * height / (te - ts);

    printf("%dx%d: T=%f[msec.], %f[MP/s]\n", width, height, (te - ts) * 1000.0, speed);

    save_image(OUTPUT_FILE, width, height, output);

    free(output);
    free(input);

    return EXIT_SUCCESS;
}

//----------------------------------------------------------------------
uchar *load_image(const char *fname, int *width, int *height)
{
    pixmap_t pixmap;
    uchar *image = NULL;

    if (pixmap_load_pnmfile(&pixmap, fname) == EXIT_FAILURE) {
	if (errno) {	/* system error */
	    perror(fname);
	} else {	/* unacceptable file format */
	    fprintf(stderr, "%s: Not a ppm, pgm, or pbm file\n", fname);
	}
	exit(EXIT_FAILURE);
    }

    if ((image = MALLOC(pixmap.width * pixmap.height, uchar)) != NULL) {
	int i, j;

	*width  = pixmap.width ;
	*height = pixmap.height;

	// convert image from pixmap to graymap
#pragma omp parallel for private(i,j) collapse(2)
	for (j = 0; j < *height; j++) {
	    for (i = 0; i < *width; i++) {
		pixel_t pixel;
		pixmap_get_pixel(&pixmap, &pixel, i, j);
		image[i + j * *width] = RGB2GRAY(pixel_get_r(pixel),
						 pixel_get_g(pixel),
						 pixel_get_b(pixel));
	    }
	}
    }

    pixmap_destroy(&pixmap);

    return image;
}

//----------------------------------------------------------------------
void save_image(const char *fname, int width, int height, uchar *image)
{
    FILE *fp = NULL;

    if ((fp = fopen(fname, "wb")) == NULL) {
	perror(fname);
	exit(EXIT_FAILURE);
    }

    fprintf(fp, "P5\n%d %d\n255\n", width, height);
    fwrite(image, sizeof(uchar), width * height, fp);

    fclose(fp);

    return;
}

//----------------------------------------------------------------------
void sobel_filter(int width, int height, uchar *input, uchar *output)
{
    int ii, jj;

#pragma omp parallel for private(ii,jj) collapse(2)
    for (jj = 0; jj < height; jj++) {
	for (ii = 0; ii < width; ii++) {
	    int sobel, sobel_h, sobel_v;
	    int ip = ii + 1, im = ii - 1,
		jp = jj + 1, jm = jj - 1;

	    if (ip > width  - 1) ip = width  - 1;
	    if (jp > height - 1) jp = height - 1;
	    if (im < 0) im = 0;
	    if (jm < 0) jm = 0;

	    sobel_h =     INPUT(im,jm)                    -     INPUT(ip,jm) +
		      2 * INPUT(im,jj)                    - 2 * INPUT(ip,jj) +
			  INPUT(im,jp)                    -     INPUT(ip,jp);
	    sobel_v =     INPUT(im,jm) + 2 * INPUT(ii,jm) +     INPUT(ip,jm) -
			  INPUT(im,jp) - 2 * INPUT(ii,jp) -     INPUT(ip,jp);
//	    sobel   = hypotf(sobel_h, sobel_v);
	    sobel   = abs(sobel_h) + abs(sobel_v);

	    OUTPUT(ii,jj) = MIN(0xff, sobel);
	}
    }

    return;
}
