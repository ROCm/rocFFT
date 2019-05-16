#!/usr/bin/env python
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

import sys
import argparse
import subprocess
import itertools
import re
import os
import math
from threading import Timer, Thread
import thread, time
from platform import system
import numpy as np

from datetime import datetime

import errorHandler
from fftPerformanceTesting import *
from performanceUtility import timeout, log, generate235Radices

#Todo list:
# - more error handling
# - more tests for relative path

TIMOUT_VAL = 900 #In seconds
WARNING_LOG_MAX_ENTRY = 500
MIN_GFLOPS_TO_COMPARE = 10
MAX_RERUN_NUM = 128

#layoutvalues = ['cp', 'ci']
placevalues = ['in', 'out']
precisionvalues = ['single', 'double']

pow10 = '1-9,10-90:10,100-900:100,1000-9000:1000,10000-90000:10000,100000-900000:100000,1000000-9000000:1000000'

parser = argparse.ArgumentParser(description='Measure performance of the rocFFT library')
parser.add_argument('--device',
    dest='device', default='0',
    help='device(s) to run on; may be a comma-delimited list. choices are (default gpu)')
parser.add_argument('-b', '--batchsize',
    dest='batchSize', default='1',
    help='number of FFTs to perform with one invocation of the client. the special value \'adapt\' may be used to adjust the batch size on a per-transform basis to the maximum problem size possible on the device. (default 1)'.format(pow10))
parser.add_argument('-a', '--adaptivemax',
    dest='constProbSize', default='-1',
    help='Max problem size that you want to maintain across the invocations of client with different lengths. This is adaptive and adjusts itself automtically.'.format(pow10))
parser.add_argument('-x', '--lengthx',
    dest='lengthx', default='1',
    help='length(s) of x to test; must be factors of 1, 2, 3, or 5 with rocFFT; may be a range or a comma-delimited list. e.g., 16-128 or 1200 or 16,2048-32768 (default 1)')
parser.add_argument('-y', '--lengthy',
    dest='lengthy', default='1',
    help='length(s) of y to test; must be factors of 1, 2, 3, or 5 with rocFFT; may be a range or a comma-delimited list. e.g., 16-128 or 1200 or 16,32768 (default 1)')
parser.add_argument('-z', '--lengthz',
    dest='lengthz', default='1',
    help='length(s) of z to test; must be factors of 1, 2, 3, or 5 with rocFFT; may be a range or a comma-delimited list. e.g., 16-128 or 1200 or 16,32768 (default 1)')
parser.add_argument('-reps',
    dest='reps', default='10',
    help='Number of repetitions (default 10)')
parser.add_argument('-prime_factor', '--prime_factor',
    dest='prime_factor', default='2',
    help='only test the prime factors within the specified range of lengthx/y/z. Select from 2,3,5, and 7. Example: -prime_factor 2,3')
parser.add_argument('-test_count', '--test_count',
    dest='test_count', default='100',
    help='Number of tests to perform')
parser.add_argument('--problemsize',
    dest='problemsize', default=None)
#    help='additional problems of a set size. may be used in addition to lengthx/y/z. each indicated problem size will be added to the list of FFTs to perform. should be entered in AxBxC:D format. A, B, and C indicate the sizes of the X, Y, and Z dimensions (respectively). D is the batch size. All values except the length of X are optional. may enter multiple in a comma-delimited list. e.g., 2x2x2:32768 or 256x256:100,512x512:256')
parser.add_argument('-i', '--inputlayout',
    dest='inputlayout', default='0',
    help=' 0. interleaved (default) 1. planar 2. real  3. hermitian interleaved 4. hermitian planar' )
parser.add_argument('-o', '--outputlayout',
    dest='outputlayout', default='0',
    help=' 0. interleaved (default) 1. planar 2. real  3. hermitian interleaved 4. hermitian planar' )
parser.add_argument('--placeness',
    dest='placeness', default='in',
    help='Choices are ' + str(placevalues) + '. in = in place, out = out of place (default in)')
parser.add_argument('-r', '--precision',
    dest='precision', default='single',
    help='Choices are ' + str(precisionvalues) + '. (default single)')
parser.add_argument('--label',
    dest='label', default=None,
    help='a label to be associated with all transforms performed in this run. if LABEL includes any spaces, it must be in \"double quotes\". note that the label is not saved to an .ini file. e.g., --label cayman may indicate that a test was performed on a cayman card or --label \"Windows 32\" may indicate that the test was performed on Windows 32')
parser.add_argument('--ref-file',
    dest='refFilename', default=None,
    help='The reference results file to compare with.')
parser.add_argument('--ref-tol',
    dest='refTol', default='0.05',
    help='The reference gflops tolerance, default 5%%.')
parser.add_argument('--tablefile',
    dest='tableOutputFilename', default=None,
    help='save the results to a plaintext table with the file name indicated. this can be used with plotPerformance.py to generate graphs of the data (default: table prints to screen)')
parser.add_argument('--mute', action="store_true", help='no print')
parser.add_argument('--client-prefix',
    dest='client_prefix', default='./',
    help='Path where the library client is located (default current directory)')
parser.add_argument('--rerun',
    dest='rerun', default=None,
    help='rerun test from *.csv result file')

args = parser.parse_args()

label = str(args.label)

# todo: change the log dir, especially for rerun case
if not os.path.exists('perfLog'):
    os.makedirs('perfLog')
logfile = os.path.join('perfLog', (label+'-'+'fftMeasurePerfLog.txt'))

def printLog(txt):
    if not args.mute:
        print txt
    log(logfile, txt)

printLog("=========================MEASURE PERFORMANCE START===========================")
printLog("Process id of Measure Performance:"+str(os.getpid()))

currCommandProcess = None

rerun_args = ''
rerun_index = -1
rerun_file = str(args.rerun)
if args.rerun:
    if (not rerun_file) or (not os.path.isfile(rerun_file)):
        printLog('ERROR: invalid file/path for --rerun option.')
        quit()
    else:
        with open(rerun_file, 'r') as input:
            for line in input:
                if line.startswith('#Cmd:'):
                    rerun_args = line.strip().split('.py')[1] #todo: better err handling
                    rerun_args = rerun_args.strip().split(' ')
                    break

        if not "--tablefile" in rerun_args :
            printLog('ERROR: --rerun option, need explicitly specified --tablefile file.')
            quit()

        for i in range(MAX_RERUN_NUM):
            next_file =  rerun_file[:-4] + "_r" + str(i) + ".csv" #support csv file only for now
            if not os.path.isfile(next_file):
                rerun_args = [arg.replace(rerun_file, next_file) for arg in rerun_args]
                #print rerun_args
                args = parser.parse_args(rerun_args)
                args.label = os.path.basename(next_file)[:-4]
                rerun_index = i
                break
            if i >= MAX_RERUN_NUM-1:
                printLog('ERROR: --rerun option, too many files.')
                quit()

args.library = 'rocFFT'

if args.tableOutputFilename != None and args.refFilename != None:
    if args.tableOutputFilename == args.refFilename:
        printLog('ERROR: tablefile and ref-file are the same.')
        quit()

printLog('Executing measure performance for label: '+str(label))


#This function is defunct now
@timeout(1, "fileName") # timeout is 5 minutes, 5*60 = 300 secs
def checkTimeOutPut2(args):
    global currCommandProcess
    #ret = subprocess.check_output(args, stderr=subprocess.STDOUT)
    #return ret
    currCommandProcess = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    printLog("Curr Command Process id = "+str(currCommandProcess.pid))
    ret = currCommandProcess.communicate()
    if(ret[0] == None or ret[0] == ''):
        errCode = currCommandProcess.poll()
        raise subprocess.CalledProcessError(errCode, args, output=ret[1])
    return ret[0]


#Spawns a separate thread to execute the library command and wait for that thread to complete
#This wait is of 900 seconds (15 minutes). If still the thread is alive then we kill the thread
def checkTimeOutPut(args):
    t = None
    global currCommandProcess
    global stde
    global stdo
    stde = None
    stdo = None
    def executeCommand():
        global currCommandProcess
        global stdo
        global stde
        try:
            stdo, stde = currCommandProcess.communicate()
            printLog('stdout:\n'+str(stdo).replace('\n', '\n                          '))
            printLog('stderr:\n'+str(stde).replace('\n', '\n                          '))
        except:
            printLog("ERROR: UNKNOWN Exception - +checkWinTimeOutPut()::executeCommand()")

    currCommandProcess = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
    thread = Thread(target=executeCommand)
    thread.start()
    thread.join(TIMOUT_VAL) #wait for the thread to complete
    if thread.is_alive():
        printLog('ERROR: Killing the process - terminating thread because it is taking too much of time to execute')
        currCommandProcess.kill()
        printLog('ERROR: Timed out exception')
        raise errorHandler.ApplicationException(__file__, errorHandler.TIME_OUT)
    if stdo == "" or stdo==None:
        errCode = currCommandProcess.poll()
        printLog('ERROR: @@@@@Raising Called processor exception')
        raise subprocess.CalledProcessError(errCode, args, output=stde)
    return stdo

#turn pow10 into its range list
if args.batchSize.count('pow10'):
    args.batchSize = pow10

#split up comma-delimited lists
args.batchSize = args.batchSize.split(',')
args.constProbSize = int(args.constProbSize.split(',')[0])
args.device = args.device.split(',')
args.lengthx = args.lengthx.split(',')
args.lengthy = args.lengthy.split(',')
args.lengthz = args.lengthz.split(',')
args.prime_factor = args.prime_factor.split(',')
if args.problemsize:
    args.problemsize = args.problemsize.split(',')
args.inputlayout = args.inputlayout.split(',')
args.outputlayout = args.outputlayout.split(',')
args.placeness = args.placeness.split(',')
args.precision = args.precision.split(',')



printLog('Executing for label: '+str(args.label))
#check parameters for sanity

# batchSize of 'max' must not be in a list (does not get on well with others)
#if args.batchSize.count('max') and len(args.batchSize) > 1:
if ( args.batchSize.count('max') or args.batchSize.count('adapt') )and len(args.batchSize) > 1:
    printLog('ERROR: --batchsize max must not be in a comma delimited list')
    quit()


# in case of an in-place transform, input and output layouts must be the same (otherwise: *boom*)
#for n in args.placeness:
#    if n == 'in' or n == 'inplace':
#        if len(args.inputlayout) > 1 or len(args.outputlayout) > 1 or args.inputlayout[0] != args.outputlayout[0]:
#            printLog('ERROR: if transformation is in-place, input and output layouts must match')
#            quit()

# check for valid values in precision
for n in args.precision:
    if n != 'single' and n != 'double':
        printLog('ERROR: invalid value for precision')
        quit()

def isPrime(n):
    import math
    n = abs(n)
    i = 2
    while i <= math.sqrt(n):
        if n%i == 0:
            return False
        i += 1
    return True

def findFactors(number):
    iter_space = range(1, number+1)
    prime_factor_list = []
    for curr_iter in iter_space:
        if isPrime(curr_iter) == True:
            #print 'curr_iter_prime: ', curr_iter
            if number%curr_iter == 0:
                prime_factor_list.append(curr_iter)
    return prime_factor_list


#Type : Function
#Input: num, a number which we need to factorize
#Return Type: list
#Details: This function returns only the prime factors on an input number
#         e.g: input: 20, returns: [2,2,5]
#              input: 32, returns: [2,2,2,2,2]
def factor(num):
    if num == 1:
        return [1]
    i = 2
    limit = num**0.5
    while i <= limit:
        if num % i == 0:
            ret = factor(num/i)
            ret.append(i)
            return ret
        i += 1
    return [num]

def validateFactors(flist):
    ref_list = [1,2,3,5]
    if flist==ref_list:
        return True
    if len(flist) > len(ref_list):
        return False
    for felement in flist:
        if ref_list.count(felement) != 1:
            return False
    return True

#Type : Function
#Input: num, a number which we need to validate for 1,2,3 or 5 factors
#Return Type: boolean
#Details: This function validates an input number for its prime factors
#         If factors has number other than 1,2,3 or 5 then return false else return true
#         e.g: input: 20, returns: True
#              input: 28, returns: False
def validate_number_for_1235(num):
    if num == 0:
        return True
    set1235 = set([1,2,3,5])
    setPrimeFactors = set(factor(num))
    setPrimeFactors = setPrimeFactors | set1235 #performed union of two sets
    #if still the sets are same then we are done!!!
    #else we got few factors other than 1,2,3 or 5 and we should invalidate
    #the input number
    if setPrimeFactors ==  set1235:
        return True
    return False


def getValidNumbersInRange(rlist):
    valid_number_list = []
    for relement in rlist:
        prime_factors = findFactors(relement)
        if validateFactors(prime_factors) == True:
            valid_number_list.append(relement)
    return valid_number_list

def get_next_num_with_1235_factors(start):
    start+=1
    while not validateFactors(findFactors(start)):
        start+=1
    return start


def check_number_for_1235_factors(number):
    #printLog('number:'+ number)
    factors = findFactors(number)
    #printLog('factors:'+ factors)
    if not validateFactors(factors):
        printLog("ERROR: --{0} must have only 1,2,3,5 as factors")
        return False
    return True



def check_for_1235_factors(values, option):
    #print 'values: ', values
    for n in values:
        for m in n.replace('-',',').split(','):
            if not validate_number_for_1235(int(m)):
                print 'ERROR: --{0} must specify number with only 1,2,3,5 as factors'.format(option)
                quit()
            #print 'Valid number for :',option,':', m


if args.library == 'rocFFT':
    check_for_1235_factors(args.lengthx, 'lengthx')
    check_for_1235_factors(args.lengthy, 'lengthy')
    check_for_1235_factors(args.lengthz, 'lengthz')



if not os.path.isfile(args.client_prefix+executable(args.library)):
    printLog("ERROR: Could not find client named {0}".format(executable(args.library)))
    quit()


def get235RadicesNumberInRange(minimum, maximum):
    if minimum == 0 and maximum == 0:
        return [0]
    numbers = generate235Radices(maximum)
    minIndex = numbers.index(minimum)
    maxIndex = numbers.index(maximum)
    return numbers[minIndex:maxIndex+1]

#expand ranges
class Range:
    def __init__(self, ranges, defaultStep='+1'):
        self.expanded = []
        for thisRange in ranges:
            if thisRange != 'max' and thisRange != 'adapt' :
                if thisRange.count(':'):
                    self._stepAmount = thisRange.split(':')[1]
                else:
                    self._stepAmount = defaultStep
                thisRange = thisRange.split(':')[0]

                if self._stepAmount.count('x'):
                    self._stepper = '_mult'
                    self._stepAmount = self._stepAmount.lstrip('+x')
                    self._stepAmount = int(self._stepAmount)
                elif self._stepAmount.count('l'):
                    self._stepper = '_next_num_with_1235_factor'
                    self._stepAmount = 0
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

           # _thisRangeExpanded = []
            if thisRange == 'max':
                self.expanded = self.expanded + ['max']
            elif thisRange == 'adapt':
                self.expanded = self.expanded + ['adapt']
            elif self.begin == 0 and self._stepper == '_mult':
                self.expanded = self.expanded + [0]
            else:
                if self._stepper == '_next_num_with_1235_factor':
                    self.expanded = self.expanded + get235RadicesNumberInRange(self.current, self.end)
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

    def _next_num_with_1235_factor(self):
        self.current = get_next_num_with_1235_factors(self.current)


args.batchSize = Range(args.batchSize).expanded
args.lengthx = Range(args.lengthx, 'l').expanded
args.lengthy = Range(args.lengthy, 'l').expanded
args.lengthz = Range(args.lengthz, 'l').expanded


def create_prime_factors(args,input_list):
  powers2=[1]
  powers3=[1]
  powers5=[1]
  powers7=[1]
  if '2' in args.prime_factor:
    powers2+=[2**x for x in range(1,int(math.floor(math.log(max(input_list),2)+1)))]
  if '3' in args.prime_factor:
    powers3+=[3**x for x in range(1,int(math.floor(math.log(max(input_list),3)+1)))]
  if '5' in args.prime_factor:
    powers5+=[5**x for x in range(1,int(math.floor(math.log(max(input_list),5)+1)))]
  if '7' in args.prime_factor:
    powers7+=[7**x for x in range(1,int(math.floor(math.log(max(input_list),7)+1)))]


  xlist=[]
  for i in powers2:
    for j in powers3:
      for k in powers5:
        for l in powers7:
          dummy=int(i)*int(j)*int(k)*int(l)
          if(dummy<=max(input_list)) and (dummy>=min(input_list)):
            xlist.append(dummy)

  xlist=sorted(xlist)
  xlist=xlist[:int(args.test_count)] #snafu
  return xlist

args.lengthx=create_prime_factors(args,args.lengthx)
args.lengthy=create_prime_factors(args,args.lengthy)
args.lengthz=create_prime_factors(args,args.lengthz)

#expand problemsizes ('XxYxZ:batch')
#print "args.problemsize--1-->", args.problemsize
if args.problemsize and args.problemsize[0] != 'None':
    i = 0
    while i < len(args.problemsize):
        args.problemsize[i] = args.problemsize[i].split(':')
        args.problemsize[i][0] = args.problemsize[i][0].split('x')
        i = i+1


#create the problem size combinations for each run of the client
# A: This part creats a product of all possible combinations. Too many cases in 2/3D
#problem_size_combinations = itertools.product(args.lengthx, args.lengthy, args.lengthz, args.batchSize)
#problem_size_combinations = list(itertools.islice(problem_size_combinations, None))

if args.lengthy[0]==1:
  args.lengthy=[1]*len(args.lengthx)
if args.lengthz[0]==1:
  args.lengthz=[1]*len(args.lengthx)

dummy=[args.batchSize[0]]*len(args.lengthx)
problem_size_combinations=zip(args.lengthx,args.lengthy,args.lengthz,dummy)

#print "args.problemsize--2-->", args.problemsize
#add manually entered problem sizes to the list of FFTs to crank out
manual_test_combinations = []
if args.problemsize and args.problemsize[0] != 'None':
    for n in args.problemsize:
        x = []
        y = []
        z = []
        batch = []

        x.append(int(n[0][0]))

        if len(n[0]) >= 2:
            y.append(int(n[0][1]))
        else:
            y.append(1)

        if len(n[0]) >= 3:
            z.append(int(n[0][2]))
        else:
            z.append(1)

        if len(n) > 1:
            batch.append(int(n[1]))
        else:
            batch.append(1)

        combos = itertools.product(x, y, z, batch)
        combos = list(itertools.islice(combos, None))
        for n in combos:
            manual_test_combinations.append(n)
        # manually entered problem sizes should not be plotted (for now). they may still be output in a table if requested


problem_size_combinations = problem_size_combinations + manual_test_combinations

#create final list of all transformations (with problem sizes and transform properties)
test_combinations = itertools.product(problem_size_combinations, args.device, args.inputlayout, args.outputlayout, args.placeness, args.precision)
test_combinations = list(itertools.islice(test_combinations, None))
test_combinations = [TestCombination(params[0][0], params[0][1], params[0][2], params[0][3], params[1], params[2], params[3], params[4], params[5], args.label) for params in test_combinations]

#print("lenghtx= ",test_combinations[0].x)
#print("lenghty= ",test_combinations[0].y)
#print("lenghtz= ",test_combinations[0].z)
#print("placeness= ",test_combinations[0].placeness)

#turn each test combination into a command, run the command, and then stash the gflops
gflops_result = [] # this is where we'll store the results for the table

#open output file and write the header

if args.tableOutputFilename == None:
    args.tableOutputFilename = 'rocFFT_' + 'x_'+ str(args.lengthx[0]) + '_y_'+str(args.lengthy[0])+'_z_'+str(args.lengthz[0])+'_'+str(args.precision[0])+ '_'+datetime.now().isoformat().replace(':','.') + '.txt'
else:
   if os.path.isfile(args.tableOutputFilename):
       oldname = args.tableOutputFilename
       args.tableOutputFilename = args.tableOutputFilename + datetime.now().isoformat().replace(':','.')
       message = 'A file with the name ' + oldname + ' already exists. Changing filename to ' + args.tableOutputFilename
       printLog(message)


printLog('table header---->'+ str(tableHeader))

table = open(args.tableOutputFilename, 'w')
table.write('#Do not change any content of this file except adding comments!!!\n')
table.write('#\n')
table.write('#Timestamp: ' + str(datetime.now()) + '\n')
table.write('#\n')
if rerun_args:
    table.write('#From --rerun\n')
    table.write('#Cmd: python ' + str(sys.argv[0]) + ' ' + str(' '.join(rerun_args)) + '\n')
else:
    table.write('#Cmd: python ' + str(' '.join(sys.argv)) + '\n')
table.write('#\n')
table.write(tableHeader + '\n')
table.flush()
if args.constProbSize == -1:
   args.constProbSize = maxBatchSize(1, 1, 1, args.inputlayout[0], args.precision[0], '-' + args.device[0])
args.constProbSize = int(args.constProbSize)


printLog('Total combinations =  '+str(len(test_combinations)))

vi = 0
for params in test_combinations:
    if vi>=int(args.test_count):
      break
    vi = vi+1
    printLog("-----------------------------------------------------")
    printLog('preparing command: '+ str(vi))
    device = params.device
    lengthx = str(params.x)
    lengthy = str(params.y)
    lengthz = str(params.z)
    inlayout=str(params.inlayout)
    outlayout=str(params.outlayout)
    client_prefix=str(args.client_prefix)


    if params.batchsize == 'max':
        batchSize = maxBatchSize(lengthx, lengthy, lengthz, params.inlayout, params.precision, '-' + device)
    elif params.batchsize == 'adapt':
        batchSize = str(args.constProbSize/(int(lengthx)*int(lengthy)*int(lengthz)))
    else:
        batchSize = str(params.batchsize)

    if params.placeness == 'inplace' or params.placeness == 'in':
        placeness = ''
    elif params.placeness == 'outofplace' or params.placeness == 'out':
        placeness = '-o'
    else:
        printLog('ERROR: invalid value for placeness when assembling client command')

    if params.precision == 'single':
        precision = ''
    elif params.precision == 'double':
        precision = '--double'
    else:
        printLog('ERROR: invalid value for precision when assembling client command')

    transformType = '0'
    if (inlayout == '2' and (outlayout == '3' or outlayout == '4')):
        transformType = '2'
    elif (outlayout == '2' and (inlayout == '3' or outlayout == '4')):
        transformType = '3'

    #set up arguments here
    arguments = [client_prefix+ executable(args.library),
                 '--device ' + device,
                 '-x', lengthx,
                 '-y', lengthy,
                 '-z', lengthz,
                 '--batchSize', batchSize,
                 '-t', transformType,
                 '--inArrType', inlayout,
                 '--outArrType',outlayout,
                 placeness,
                 precision,
                 '-p', args.reps]


    writeline = True
    try:
        arguments=' '.join(arguments)
        printLog('Executing Command: '+str(arguments))
        output = checkTimeOutPut(arguments)
        output = output.split(os.linesep);
        printLog('Execution Successfull\n')

    except errorHandler.ApplicationException as ae:
        writeline = False
        printLog('ERROR: Command is taking too much of time '+ae.message+'\n'+'Command: \n'+str(arguments))
        continue
    except subprocess.CalledProcessError as clientCrash:
        print 'Command execution failure--->'
        writeline = False
        printLog('ERROR: client crash. Please report the following error message (with rocFFT error code, if given, and the parameters used to invoke measurePerformance.py) \n'+clientCrash.output+'\n')
        printLog('IN ORIGINAL WE CALL QUIT HERE - 1\n')
        continue

    for x in output:
        if x.count('out of memory'):
            writeline = False
            printLog('ERROR: Omitting line from table - problem is too large')
            printLog('ERROR: Omitting line from table - problem is too large')

    if writeline:
        try:
            output = itertools.ifilter( lambda x: x.count('gflops'), output)

            output = list(itertools.islice(output, None))
            thisResult = re.search('\d+\.*\d*e*-*\d*$', output[-1])
            thisResult = float(thisResult.group(0))
            gflops_result.append(thisResult)

            thisResult = ('{:11d}'.format(params.x), '{:11d}'.format(params.y), '{:11d}'.format(params.z),\
                '{:>11s}'.format(batchSize), '{:>7s}'.format(params.device), \
                '{:>6s}'.format(params.inlayout), '{:>7s}'.format(params.outlayout), \
                '{:>6s}'.format(params.placeness), '{:>10s}'.format(params.precision), \
                '{:>12s}'.format(params.label), '{:>11.3f}'.format(thisResult))

            outputRow = ''
            for x in thisResult:
                outputRow = outputRow + str(x) + ','
            #outputRow = outputRow.rstrip(',')
            table.write(outputRow + '\n')
            table.flush()

        except:
			printLog('ERROR: Exception occurs in GFLOP parsing')
    else:
        if(len(output) > 0):
            if output[0].find('nan') or output[0].find('inf'):
                printLog( 'WARNING: output from client was funky for this run. skipping table row')
            else:
                prinLog('ERROR: output from client makes no sense')
                printLog(str(output[0]))
                printLog('IN ORIGINAL WE CALL QUIT HERE - 2\n')
        else:
            prinLog('ERROR: output from client makes no sense')
            #quit()

if args.refFilename != None:
    printLog("-----------------------------------------------------")
    printLog("Enabled reference comparison")
    refResults = open(args.refFilename, 'r')
    refResultsContents = refResults.read()
    refResultsContents = refResultsContents.rstrip().split('\n')

    raw_data = []
    for line in refResultsContents:
        if not (line.startswith('#') or len(line.strip()) == 0):
            raw_data.append(line.split('#')[0].rstrip(', '))

    printLog("          index"+str(tableHeader).replace("     GFLOPS", "  GFLOPS ref vs tested  relative_err"))
    failedCount = 0
    totalCount = len(gflops_result)
    for idx, row in enumerate(raw_data):
        ref_gflops = float(row[row.rfind(',')+1:]) # assume the last col is GFLOPS
        if (idx < totalCount and np.less(MIN_GFLOPS_TO_COMPARE, ref_gflops) ):
            if np.less(gflops_result[idx], ref_gflops):
                relative_error = abs(ref_gflops - gflops_result[idx])/gflops_result[idx]
                if np.greater(relative_error, float(args.refTol)):
                    printLog("Warning: " + '{:>6d}'.format(idx+1) + row +
                             "," + '{:>11.3f}'.format(gflops_result[idx]) +
                             "," + '{:12.2%}'.format(-relative_error))
                    failedCount+=1
            if failedCount>= WARNING_LOG_MAX_ENTRY:
                printLog("Too many failed cases...")
                break

    printLog("\nTotal number of samples " + str(totalCount) +
             ", passing rate " + '{:.2%}'.format((totalCount-failedCount)/totalCount) +
             ", with tolerance " + '{:.2%}'.format(float(args.refTol)))

if rerun_args:
    printLog("-----------------------------------------------------")
    printLog("Rerun auto plotting...")

    plot_output_file = ''
    plot_cmd = "python plotPerformance.py -x x -y gflops -d " + rerun_file
    if rerun_index > -1:
        for i in range(rerun_index+1):
                next_file =  rerun_file[:-4] + "_r" + str(i) + ".csv" #support csv file only for now
                if os.path.isfile(next_file):
                    plot_cmd +=  " -d " + next_file
                if i == rerun_index:
                    plot_output_file = next_file.replace(".csv", ".png")
                    plot_cmd +=  " --outputfile " + plot_output_file
    subprocess.check_call(plot_cmd, shell=True)
    printLog("Plotted to file " + plot_output_file + '.')

printLog("=========================MEASURE PERFORMANCE ENDS===========================\n")

