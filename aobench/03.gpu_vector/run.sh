#!/bin/sh
# $Id: run.sh,v 1.1.1.1 2020/07/29 00:00:00 seiji Exp seiji $

usage() {
    echo "Usage:: `basename $0` [-d cpu|gpu] [-f float|double] [-m native|system] [-s #sub] [-t #ao]" 1>&2
    exit ${1:-0}
}

# defaults
dtype=default
rtype=double
math=system
n_sub=2
n_ao=8

while getopts d:f:hm:s:t: OPT
do
    case $OPT in
    d)
	dtype=$OPTARG
	;;
    f)
	rtype=$OPTARG
	;;
    h)
	usage
	;;
    m)
	math=$OPTARG
	;;
    s)
	n_sub=$OPTARG
	;;
    t)
	n_ao=$OPTARG
	;;
    \?)
	usage 1
	;;
    esac
done

shift $((OPTIND - 1))

exec_bin=ao.exe
export LIBPIXMAP_X11_RES_CLASS=${exec_bin}
export LIBPIXMAP_X11_RES_NAME=${exec_bin}
export LIBPIXMAP_X11_ASYNC=true

( make BIN=$exec_bin clean ) > /dev/null 2>&1
( make BIN=$exec_bin DEVICE=$dtype TYPE=$rtype MATH=$math SUB_SAMPLE=$n_sub AO_SAMPLE=$n_ao ) > /dev/null 2>&1

exec ./$exec_bin
