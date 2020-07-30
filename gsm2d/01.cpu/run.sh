#!/bin/sh
# $Id: run.sh,v 1.1.1.1 2020/07/29 00:00:00 seiji Exp seiji $

usage() {
    echo "Usage:: `basename $0` [-f float|double] [-i #data]" 1>&2
    exit ${1:-0}
}

# defaults
rtype=double
input=013

while getopts f:hi: OPT
do
    case $OPT in
    f)
	rtype=$OPTARG
	;;
    h)
	usage
	;;
    i)
	input=$OPTARG
	;;
    \?)
	usage 1
	;;
    esac
done

shift $((OPTIND - 1))

exec_bin=gsm2d.exe
export LIBPIXMAP_X11_RES_CLASS=${exec_bin}
export LIBPIXMAP_X11_RES_NAME=${exec_bin}
export LIBPIXMAP_X11_ASYNC=true

( make BIN=$exec_bin clean ) > /dev/null 2>&1
( make BIN=$exec_bin TYPE=$rtype DATA=$input ) > /dev/null 2>&1

exec ./$exec_bin
