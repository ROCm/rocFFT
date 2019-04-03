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

def load_short_test_suite(measure_cmd, label, file_list):
    subprocess.check_call(measure_cmd + label + " -x 2-16777216                     -b adapt -prime_factor 2                          " + file_list[0], shell=True)
    subprocess.check_call(measure_cmd + label + " -x 2-4096      -y 2-4096          -b adapt -prime_factor 2                          " + file_list[1], shell=True)
    subprocess.check_call(measure_cmd + label + " -x 2-256       -y 2-256  -z 2-256 -b adapt -prime_factor 2                          " + file_list[2], shell=True)
                                                                                                           
    subprocess.check_call(measure_cmd + label + " -x 2-16777216                     -b adapt -prime_factor 2           --placeness out" + file_list[3], shell=True)
    subprocess.check_call(measure_cmd + label + " -x 5-9765625                      -b adapt -prime_factor 5                          " + file_list[4], shell=True)
    subprocess.check_call(measure_cmd + label + " -x 128-4194304                             -prime_factor 2 -i 2 -o 3                " + file_list[5], shell=True) # TODO: test with "-x 128-4194304 -b adapt" after fixing real fft 
    subprocess.check_call(measure_cmd + label + " -x 81-177147                               -prime_factor 3 -i 2 -o 4 --placeness out" + file_list[6], shell=True) # TODO: test with "-x 81-1594323 -b adapt" after fixing real fft
    subprocess.check_call(measure_cmd + label + " -x 2-4096      -y 2-4096          -b 20    -prime_factor 2 -i 3 -o 2 --placeness out" + file_list[7], shell=True) # TODO: test with "-b adapt" after fixing real fft
                                                                                                           
    subprocess.check_call(measure_cmd + label + " -x 2-16777216                     -b adapt -prime_factor 2 -r double                " + file_list[8], shell=True)
    subprocess.check_call(measure_cmd + label + " -x 2-4096      -y 2-4096          -b adapt -prime_factor 2 -r double                " + file_list[9], shell=True)
    subprocess.check_call(measure_cmd + label + " -x 2-256       -y 2-256  -z 2-256 -b adapt -prime_factor 2 -r double                " + file_list[10], shell=True)
                                                                                                           
    subprocess.check_call(measure_cmd + label + " -x 256-16777216                   -b adapt -prime_factor 2 -r double --placeness out" + file_list[11], shell=True)
    subprocess.check_call(measure_cmd + label + " -x 256-4194304                    -b 50    -prime_factor 2 -r double -i 2 -o 3      " + file_list[12], shell=True) # TODO: test with "-b adapt" after fixing real fft



parser = argparse.ArgumentParser(description='rocFFT performance test suite')
parser.add_argument('-t', '--type',
    dest='type', default='full',
    help='run suite with full or short suite(default full)')
parser.add_argument('-r', '--ref_dir',
    dest='ref_dir', default='./',
    help='specify the reference results dirctory(default ./)')
parser.add_argument('-w','--work_dir',
    dest='work_dir', default='./',
    help='specify the current working results dirctory(default ./)')
parser.add_argument('-g', '--gen_ref', action="store_true", help='generate reference')    

args = parser.parse_args()

elapsed_time = timer()

measure_cmd = "python measurePerformance.py"
file_name_index_list = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'd0', 'd1', 'd2', 'd3', 'd4']

if args.gen_ref:

    if not os.path.exists(args.ref_dir):
        os.mkdir( args.ref_dir, 0755 )

    # backup first
    file_list = []
    for file_name_index in file_name_index_list:
        file = args.ref_dir+'short_'+file_name_index+'_ref.csv'
        if os.path.isfile(file):
            os.rename(file, file+".bak");
        file_list.append(" --tablefile "+file)

    label = " --label short_ref "
    
    load_short_test_suite(measure_cmd, label, file_list)

else:
        
    file_list = []
    for file_name_index in file_name_index_list:
        file = args.work_dir+'short_'+file_name_index+'.csv'
        ref_file = args.ref_dir+'short_'+file_name_index+'_ref.csv'
        
        if not os.path.isfile(ref_file):
            sys.exit('Error! Can not find ref file '+ref_file)
        file_list.append(" --tablefile "+file+" --ref_file "+ref_file)

    if not os.path.exists(args.work_dir):
        os.mkdir( args.work_dir, 0755 )
    
    label = " --label short "
    
    load_short_test_suite(measure_cmd, label, file_list)

elapsed_time = timer() - elapsed_time

print "Elapsed time: " + str(datetime.timedelta(seconds=elapsed_time))

