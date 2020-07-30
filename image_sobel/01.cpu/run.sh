#!/bin/sh
# $Id: run.sh,v 1.1.1.1 2020/07/30 00:00:00 seiji Exp seiji $

usage() {
    echo "Usage:: `basename $0` [-i input]" 1>&2
    exit ${1:-0}
}

# defaults
input=input/lena.pgm

while getopts hi: OPT
do
    case $OPT in
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

exec_bin=sobel.exe

( make BIN=$exec_bin clean ) > /dev/null 2>&1
( make BIN=$exec_bin INPUT=$input ) > /dev/null 2>&1

exec ./$exec_bin
