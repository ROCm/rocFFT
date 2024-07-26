#!/bin/bash

# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Use example:
# ./bench-mpi -p /rocfft/build_mpi/clients/staging -n 2

usage() { echo "Usage: $0 [-p <path_to_staging>] [-n <number of MPI procs>]" 1>&2; exit 1; }

while getopts ":p:n:l:" o; do
    case "${o}" in
        p)
            p=${OPTARG}
            ;;    
        n)
            n=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

export RED=$(tput setaf 1 :-"" 2>/dev/null)
export GREEN=$(tput setaf 2 :-"" 2>/dev/null)
export RESET=$(tput sgr0 :-"" 2>/dev/null)

# Use a saved database of tokens, otherwise obtain them from rocfft-test
if [ -f mpi_tokens.dat ]; then
    echo "Reading MPI tokens from mpi_tokens.dat!"
else
    echo "Creating MPI tokens using rocfft-test"
    echo ${p}/rocfft-test --gtest_filter=multi_gpu* --gtest_list_tests --mp_lib mpi --mp_ranks ${n} --mp_launch \"/usr/bin/mpirun -np ${n} ${p}/rocfft_mpi_worker\"
    ${p}/rocfft-test --gtest_filter=multi_gpu* --gtest_list_tests --mp_lib mpi --mp_ranks ${n} --mp_launch "/usr/bin/mpirun -np ${n} ${p}/rocfft_mpi_worker" > mpi_tokens.dat
fi

# Filtering
grep fftw mpi_tokens.dat | awk {'print $1'} | cut -d/ -f2- > tokens

rm table_mpi_results 2> /dev/null 

# Run benchmark using rocfft_mpi_worker
while IFS= read -r token; do
    echo "mpirun --np 2 rocfft_mpi_worker $token --benchmark"
    mpirun --np 2 rocfft_mpi_worker $token --benchmark >> table_mpi_results < /dev/null
done < tokens

printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
echo '                  FFT token                       Time '
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
while IFS= read -r line1 && IFS= read -r line2 <&3; do
    a=`echo $line1  | cut -d_ -f-8`
    b=`echo $line2 | grep Max | awk {'print $4" "$5'}`
    echo $GREEN $a '   ' $b   $RESET
    echo ; printf -- "-%.0s" $(seq $(tput cols)); echo $RESET
done < tokens 3< table_mpi_results