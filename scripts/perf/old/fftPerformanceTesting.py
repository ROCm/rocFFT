# #############################################################################
# Copyright (c) 2013 - present Advanced Micro Devices, Inc. All rights reserved.
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

import itertools
import re
import subprocess
import os
import sys
from datetime import datetime

# Common data and functions for the performance suite

tableHeader = '#   lengthx,    lengthy,    lengthz,      batch, device, inlay, outlay, place, precision,       label,     GFLOPS'

class TestCombination:
    def __init__(self,
                 lengthx, lengthy, lengthz, batchsize,
                 device, inlayout, outlayout, placeness, precision,                 
                 label):
        self.x = lengthx
        self.y = lengthy
        self.z = lengthz
        self.batchsize = batchsize
        self.device = device
        self.inlayout = inlayout
        self.outlayout = outlayout
        self.placeness = placeness
        self.precision = precision
        self.label = label

    def __str__(self):
        return self.x + 'x' + self.y + 'x' + self.z + ':' + self.batchsize + ', ' + self.device + ', ' + self.inlayout + '/' + self.outlayout + ', ' + self.placeness + ', ' + self.precision + ' -- ' + self.label

class GraphPoint:
    def __init__(self,
                 lengthx, lengthy, lengthz, batchsize,
				 precision, device, label,
                 gflops):
        self.x = lengthx
        self.y = lengthy
        self.z = lengthz
        self.batchsize = batchsize
        self.device = device
        self.label = label
        self.precision = precision
        self.gflops = gflops
        self.problemsize = str(int(self.x) * int(self.y) * int(self.z) * int(self.batchsize))

    def __str__(self):
        # ALL members must be represented here (x, y, z, batch, device, label, etc)
        return self.x + 'x' + self.y + 'x' + self.z + ':' + self.batchsize + ', ' + self.precision + ' precision, ' + self.device + ', -- ' + self.label + '; ' + self.gflops

class TableRow:
    # parameters = class TestCombination instantiation
    def __init__(self, parameters, gflops):
        self.parameters = parameters
        self.gflops = gflops

    def __str__(self):
        return self.parameters.__str__() + '; ' + self.gflops

def transformDimension(x,y,z):
    if int(z) != 1:
        return 3
    elif int(y) != 1:
        return 2
    elif int(x) != 1:
        return 1

def executable(library):
    if type(library) != str:
        print 'ERROR: expected library name to be a string'
        quit()

    if sys.platform != 'win32' and sys.platform != 'linux2':
        print 'ERROR: unknown operating system'
        quit()

    if library == 'rocFFT' or library == 'null':
        if sys.platform == 'win32':
            exe = 'rocfft-rider.exe'
        elif sys.platform == 'linux2':
            exe = 'rocfft-rider'
    else:
        print 'ERROR: unknown library -- cannot determine executable name'
        quit()

    return exe

def max_mem_available_in_bytes(exe, device):
    arguments = [exe, '-i', device]
    
    deviceInfo = subprocess.check_output(arguments, stderr=subprocess.STDOUT).split(os.linesep)
    deviceInfo = itertools.ifilter( lambda x: x.count('MAX_MEM_ALLOC_SIZE'), deviceInfo)
    deviceInfo = list(itertools.islice(deviceInfo, None))
    maxMemoryAvailable = re.search('\d+', deviceInfo[0])
    return int(maxMemoryAvailable.group(0))

def max_problem_size(layout, precision, device):

    if precision == 'single':
        bytes_in_one_number = 4
    elif precision == 'double':
        bytes_in_one_number = 8
    else:
        print 'max_problem_size(): unknown precision'
        quit()

    max_problem_size = pow(2,25)
    if layout == '5':
      max_problem_size = pow(2,24) # TODO: Upper size limit for real transform
    return max_problem_size

def maxBatchSize(lengthx, lengthy, lengthz, layout, precision, device):
    problemSize = int(lengthx) * int(lengthy) * int(lengthz)
    maxBatchSize = max_problem_size(layout, precision, device) / problemSize
    return str(maxBatchSize)

def create_ini_file_if_requested(args):
    if args.createIniFilename:
        for x in vars(args):
            if (type(getattr(args,x)) != file) and x.count('File') == 0:
                args.createIniFilename.write('--' + x + os.linesep)
                args.createIniFilename.write(str(getattr(args,x)) + os.linesep)
        quit()
    
def load_ini_file_if_requested(args, parser):
    if args.useIniFilename:
        argument_list = args.useIniFilename.readlines()
        argument_list = [x.strip() for x in argument_list]
        args = parser.parse_args(argument_list)
    return args

def is_numeric_type(x):
    return type(x) == int or type(x) == long or type(x) == float

def split_up_comma_delimited_lists(args):
    for x in vars(args):
        attr = getattr(args, x)
        if attr == None:
            setattr(args, x, [None])
        elif is_numeric_type(attr):
            setattr(args, x, [attr])
        elif type(attr) == str:
            setattr(args, x, attr.split(','))
    return args

class Range:
    def __init__(self, ranges, defaultStep='+1'):
        # we might be passed in a single value or a list of strings
        # if we receive a single value, we want to feed it right back
        if type(ranges) != list:
            self.expanded = ranges
        elif ranges[0] == None:
            self.expanded = [None]
        else:
            self.expanded = []
            for thisRange in ranges:
                thisRange = str(thisRange)
                if re.search('^\+\d+$', thisRange):
                    self.expanded = self.expanded + [thisRange]
                elif thisRange == 'max':
                    self.expanded = self.expanded + ['max']
                else:
                #elif thisRange != 'max':
                    if thisRange.count(':'):
                        self._stepAmount = thisRange.split(':')[1]
                    else:
                        self._stepAmount = defaultStep
                    thisRange = thisRange.split(':')[0]

                    if self._stepAmount.count('x'):
                        self._stepper = '_mult'
                    else:
                        self._stepper = '_add'
                    self._stepAmount = self._stepAmount.lstrip('+x')
                    self._stepAmount = int(self._stepAmount)

                    if thisRange.count('-'):
                        self.begin = int(thisRange.split('-')[0])
                        self.end = int(thisRange.split('-')[1])
                    else:
                        self.begin = int(thisRange.split('-')[0])
                        self.end = int(thisRange.split('-')[0])
                    self.current = self.begin

                    if self.begin == 0 and self._stepper == '_mult':
                        self.expanded = self.expanded + [0]
                    else:
                        while self.current <= self.end:
                            self.expanded = self.expanded + [self.current]
                            self._step()

                # now we want to uniquify and sort the expanded range
                self.expanded = list(set(self.expanded))
                self.expanded.sort()

    # advance current value to next
    def _step(self):
        getattr(self, self._stepper)()

    def _mult(self):
        self.current = self.current * self._stepAmount

    def _add(self):
        self.current = self.current + self._stepAmount

def expand_range(a_range):
    return Range(a_range).expanded

def decode_parameter_problemsize(problemsize):
    if not problemsize.count(None):
        i = 0
        while i < len(problemsize):
            problemsize[i] = problemsize[i].split(':')
            j = 0
            while j < len(problemsize[i]):
                problemsize[i][j] = problemsize[i][j].split('x')
                j = j+1
            i = i+1

    return problemsize
