#!/usr/bin/env python
# #############################################################################
# Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.
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
# #############################################################################

import sys
import argparse
import subprocess
import os
import datetime

from timeit import default_timer as timer

#Todo:
#  -- implement full test suite
#  -- update short test suite
#  -- add option to run with float or double
#  -- timeout for plotting
#  -- add re-plot option
#  -- configure test suite with YAML or json

FULL_SUITE_FLOAT_TEST_NUM=9
FULL_SUITE_DOUBLE_TEST_NUM=9

SHORT_SUITE_FLOAT_TEST_NUM=7
SHORT_SUITE_DOUBLE_TEST_NUM=5

def load_short_test_suite(measure_cmd, table_file_list, ref_file_list, append_options):

    file_list = []
    if not ref_file_list:
        for f in table_file_list:
            file_list.append(" --tablefile " + f)
    else:
        for i in range(len(table_file_list)):
            file_list.append(" --tablefile " + table_file_list[i] + " --ref-file "+ ref_file_list[i])

    subprocess.check_call(measure_cmd + " -x 2-16777216                     -b adapt -prime_factor 2                          " + file_list[ 0] + append_options, shell=True)
    subprocess.check_call(measure_cmd + " -x 2-4096      -y 2-4096          -b adapt -prime_factor 2                          " + file_list[ 1] + append_options, shell=True)
    subprocess.check_call(measure_cmd + " -x 2-256       -y 2-256  -z 2-256 -b adapt -prime_factor 2                          " + file_list[ 2] + append_options, shell=True)

    subprocess.check_call(measure_cmd + " -x 2-16777216                     -b adapt -prime_factor 2           --placeness out" + file_list[ 3] + append_options, shell=True)
    subprocess.check_call(measure_cmd + " -x 5-9765625                      -b adapt -prime_factor 5                          " + file_list[ 4] + append_options, shell=True)
    subprocess.check_call(measure_cmd + " -x 128-4194304                             -prime_factor 2 -i 2 -o 3                " + file_list[ 5] + append_options, shell=True) # TODO: test with "-x 128-4194304 -b adapt" after fixing real fft
    subprocess.check_call(measure_cmd + " -x 81-177147                               -prime_factor 3 -i 2 -o 4 --placeness out" + file_list[ 6] + append_options, shell=True) # TODO: test with "-x 81-1594323 -b adapt" after fixing real fft
    subprocess.check_call(measure_cmd + " -x 2-4096      -y 2-4096          -b 20    -prime_factor 2 -i 3 -o 2 --placeness out" + file_list[ 7] + append_options, shell=True) # TODO: test with "-b adapt" after fixing real fft

    subprocess.check_call(measure_cmd + " -x 2-16777216                     -b adapt -prime_factor 2 -r double                " + file_list[ 8] + append_options, shell=True)
    subprocess.check_call(measure_cmd + " -x 2-4096      -y 2-4096          -b adapt -prime_factor 2 -r double                " + file_list[ 9] + append_options, shell=True)
    subprocess.check_call(measure_cmd + " -x 2-256       -y 2-256  -z 2-256 -b adapt -prime_factor 2 -r double                " + file_list[10] + append_options, shell=True)

    subprocess.check_call(measure_cmd + " -x 256-16777216                   -b adapt -prime_factor 2 -r double --placeness out" + file_list[11] + append_options, shell=True)
    subprocess.check_call(measure_cmd + " -x 256-4194304                    -b 50    -prime_factor 2 -r double -i 2 -o 3      " + file_list[12] + append_options, shell=True) # TODO: test with "-b adapt" after fixing real fft


def plot_test_suite(plot_cmd, table_file_list, ref_file_list):

    append_options = " -x x -y gflops "

    if not ref_file_list:
        for f in table_file_list:
            subprocess.check_call(plot_cmd + " -d " + f  + " --outputfile " + f.replace(".csv", ".png") + append_options, shell=True)
    else:
        for i in range(len(table_file_list)):
            ref_file_name = ref_file_list[i][ref_file_list[i].rfind('/')+1:]
            ref_file_name = ref_file_name.replace(".csv", ".png")
            out_file_name = table_file_list[i].replace(".csv", "-vs-" + ref_file_name)
            subprocess.check_call(plot_cmd + " -d " + table_file_list[i] + " -d " + ref_file_list[i] + " --outputfile " + out_file_name + append_options, shell=True)


parser = argparse.ArgumentParser(description='rocFFT performance test suite')
parser.add_argument('-d', '--device',
    dest='device', default='0',
    help='device(s) to run on; may be a comma-delimited list.')
parser.add_argument('-t', '--type',
    dest='type', default='full',
    help='run tests with full or short suite(default full)')
parser.add_argument('-r', '--ref-dir',
    dest='ref_dir', default='./',
    help='specify the reference results dirctory(default ./)')
parser.add_argument('-w', '--work-dir',
    dest='work_dir', default='./',
    help='specify the current working results dirctory(default ./)')
parser.add_argument('-g', '--gen-ref', action="store_true", help='generate reference')
parser.add_argument('-p', '--plot', action="store_true", help='plot the results to png')
parser.add_argument('-m','--mute', action="store_true", help='no print')
parser.add_argument('--client-prefix',
    dest='client_prefix', default='./',
    help='Path where the library client is located (default current directory)')

args = parser.parse_args()

elapsed_time = timer()

measure_cmd = "python measurePerformance.py"
plot_cmd = "python plotPerformance.py"

file_name_index_list = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'd0', 'd1', 'd2', 'd3', 'd4']

append_options = ""

table_file_list = []
ref_file_list = []

args.ref_dir = os.path.join(args.ref_dir, '')
args.work_dir = os.path.join(args.work_dir, '')

if args.gen_ref:

    if not os.path.exists(args.ref_dir):
        os.mkdir( args.ref_dir, 0755 )

    # backup first
    for file_name_index in file_name_index_list:
        file = args.ref_dir+'short_'+file_name_index+'_ref.csv'
        if os.path.isfile(file):
            os.rename(file, file+".bak");
        table_file_list.append(file)

    label = " --label short_ref "

else:

    for file_name_index in file_name_index_list:
        file = args.work_dir+'short_'+file_name_index+'.csv'
        ref_file = args.ref_dir+'short_'+file_name_index+'_ref.csv'

        if not os.path.isfile(ref_file):
            sys.exit('Error! Can not find ref file '+ref_file)
        table_file_list.append(file)
        ref_file_list.append(ref_file)

    if not os.path.exists(args.work_dir):
        os.mkdir( args.work_dir, 0755 )

    label = " --label short "

append_options += label + ' --client-prefix ' + args.client_prefix
if args.mute:
    append_options += ' --mute '

load_short_test_suite(measure_cmd, table_file_list, ref_file_list, append_options)
if args.plot:
    plot_test_suite(plot_cmd, table_file_list, ref_file_list)

elapsed_time = timer() - elapsed_time

print "Elapsed time: " + str(datetime.timedelta(seconds=elapsed_time))


