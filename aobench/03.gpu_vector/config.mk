#========================================================================
# config.mk
# $Id: config.mk,v 1.1.1.1 2020/07/29 00:00:00 seiji Exp seiji $
#========================================================================
# DEVICE : OpenCL device [default|cpu|gpu]
#........................................................................
DEVICE	= default
#------------------------------------------------------------------------
# DELAY : micro sleep delay in sec. for animation (less than 1.0)
#........................................................................
#DELAY	= 0.05
#------------------------------------------------------------------------
# MATH  : math functions [native|system]
#........................................................................
MATH	= system
#------------------------------------------------------------------------
# TYPE  : data type [float|double]
#........................................................................
TYPE	= double
#------------------------------------------------------------------------
# WIDTH : width  of output graphics
# HEIGHT: height of output graphics
#........................................................................
WIDTH	= 640
HEIGHT	= 480
#------------------------------------------------------------------------
# FRAME : animation parameter (>=1)
#........................................................................
FRAME	= 256
#------------------------------------------------------------------------
# SUB_SAMPLE: number of sub sampling (>=1)
#........................................................................
SUB_SAMPLE	= 2
#------------------------------------------------------------------------
# AO_SAMPLE : number of ambient occlusion sampling (>=1)
#........................................................................
AO_SAMPLE	= 8
