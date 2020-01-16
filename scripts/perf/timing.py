#!/usr/bin/python3

# a timing script for FFTs and convolutions using OpenMP

import sys, getopt
import numpy as np
from math import *
import subprocess
import os
import re # regexp package
import shutil
import tempfile

usage = '''A timing script for rocfft

Usage:
\ttiming.py
\t\t-D <-1,1>   default: -1 (forward).  Direction of transform
\t\t-I          make transform in-place
\t\t-N <int>    number of tests per problem size
\t\t-o <string> name of output file
\t\t-R          set transform to be real/complex or complex/real
\t\t-w <string> set working directory for rocfft-rider
\t\t-d <1,2,3>  default: dimension of transform
\t\t-x <int>    minimum problem size in x direction
\t\t-X <int>    maximum problem size in x direction
\t\t-y <int>    minimum problem size in y direction
\t\t-Y <int>    maximum problem size in Y direction
\t\t-z <int>    minimum problem size in z direction
\t\t-Z <int>    maximum problem size in Z direction
\t\t-f <string> precision: float(default) or double
\t\t-b <int>    batch size
\t\t-g <int>    device number'''

def runcase(workingdir,
            length, direction, rcfft, inplace, ntrial,
            precision, nbatch, devicenum, logfilename):
    progname = "rocfft-rider"
    prog = os.path.join(workingdir, progname)
    
    cmd = []
    cmd.append(prog)

    cmd.append("--verbose")
    cmd.append("0")
    
    cmd.append("-N")
    cmd.append("10")

    
    cmd.append("--length")
    for val in length:
        cmd.append(str(val))
    
    print(precision)
    if precision == "double":
        cmd.append("--double")
        
    cmd.append("-b")
    cmd.append(str(nbatch))

    cmd.append("--device")
    cmd.append(str(devicenum))
    
    ttype = -1
    itype = ""
    otype = ""
    if rcfft:
        if (direction == -1):
            ttype = 2
            itype = 2
            otype = 3
        if (direction == 1):
            ttype = 3
            itype = 3
            otype = 2
    else:
        itype = 0
        otype = 0
        if (direction == -1):
            ttype = 0
        if (direction == 1):
            ttype = 1
    cmd.append("-t")
    cmd.append(str(ttype))

    cmd.append("--itype")
    cmd.append(str(itype))

    cmd.append("--otype")
    cmd.append(str(otype))
    
    
    print(cmd)
    print(" ".join(cmd))

    fout = tempfile.TemporaryFile(mode="w+")
    proc = subprocess.Popen(cmd, cwd=os.path.join(workingdir,"..",".."),
                            stdout=fout, stderr=fout,
                            env=os.environ.copy())
    proc.wait()
    rc = proc.returncode
    vals = []

    fout.seek(0)

    cout = fout.read()
    logfile = open(logfilename, "a")
    logfile.write(" ".join(cmd))
    logfile.write(cout)
    logfile.close()
    
    if rc == 0:
        # ferr.seek(0)
        # cerr = ferr.read()
        searchstr = "Execution gpu time: "
        for line in cout.split("\n"):
            #print(line)
            if line.startswith(searchstr):
                # Line ends with "ms", so remove that.
                ms_string = line[len(searchstr):-2]
                #print(ms_string)
                for val in ms_string.split():
                    #print(val)
                    vals.append(1e-3 * float(val))
        print("seconds: ", vals)
                        
                        
    else:
        print("\twell, that didn't work")
        print(rc)
        print(" ".join(cmd))
        return []
                
    fout.close()
    
    return vals
    

def main(argv):
    workingdir = "."
    dimension = 1
    xmin = 2
    xmax = 1024
    ymin = 2
    ymax = 1024
    zmin = 2
    zmax = 1024
    ntrial = 10
    outfilename = "timing.dat"
    direction = -1
    rcfft = False
    inplace = False
    precision = "float"
    nbatch = 1
    radix = 2
    devicenum = 0
    
    try:
        opts, args = getopt.getopt(argv,"hb:d:D:IN:o:Rw:x:X:y:Y:z:Z:f:r:g:")
    except getopt.GetoptError:
        print("error in parsing arguments.")
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h"):
            print(usage)
            exit(0)
        elif opt in ("-d"):
            dimension = int(arg)
            if not dimension in {1,2,3}:
                print("invalid dimension")
                print(usage)
                sys.exit(1)
        elif opt in ("-D"):
            if(int(arg) in [-1,1]):
                direction = int(arg)
            else:
                print("invalid direction: " + arg)
                print(usage)
                sys.exit(1)
        elif opt in ("-I"):
            inplace = True
        elif opt in ("-o"):
            outfilename = arg
        elif opt in ("-R"):
            rcfft = True
        elif opt in ("-w"):
            workingdir = arg
        elif opt in ("-N"):
            ntrial = int(arg)
        elif opt in ("-x"):
            xmin = int(arg)
        elif opt in ("-X"):
            xmax = int(arg)
        elif opt in ("-y"):
            ymin = int(arg)
        elif opt in ("-Y"):
            ymax = int(arg)
        elif opt in ("-z"):
            zmin = int(arg)
        elif opt in ("-Z"):
            zmax = int(arg)
        elif opt in ("-b"):
            nbatch = int(arg)
        elif opt in ("-r"):
            radix = int(arg)
        elif opt in ("-f"):
            if arg not in ["float", "double"]:
                print("precision must be float or double")
                print(usage)
                sys.exit(1)
            precision = arg
        elif opt in ("-g"):
            devicenum = int(arg)

            
    print("workingdir: "+ workingdir)
    print("outfilename: "+ outfilename)
    print("ntrial: " + str(ntrial))
    print("dimension: " + str(dimension))
    print("xmin: "+ str(xmin) + " xmax: " + str(xmax))
    if dimension > 1:
        print("ymin: "+ str(ymin) + " ymax: " + str(ymax))
    if dimension > 2:
        print("zmin: "+ str(zmin) + " zmax: " + str(zmax))
    print("direction: " + str(direction))
    print("real/complex FFT? " + str(rcfft))
    print("in-place? " + str(inplace))
    print("batch-size: " + str(nbatch))
    print("radix: " + str(radix))
    print("device number: " + str(devicenum))
    
    progname = "rocfft-rider"
    prog = os.path.join(workingdir, progname)
    if not os.path.isfile(prog):
        print("**** Error: unable to find " + prog)
        sys.exit(1)

    logfilename = outfilename + ".log"
    print("log filename: "  + logfilename)
    logfile = open(logfilename, "w+")
    metadatastring = "# " + " ".join(sys.argv)  + "\n"
    metadatastring += "# "
    metadatastring += "dimension"
    metadatastring += "\txlength"
    if(dimension > 1):
        metadatastring += "\tylength"
    if(dimension > 2):
        metadatastring += "\tzlength"
    metadatastring += "\tnbatch"
    metadatastring += "\tnsample"
    metadatastring += "\tsamples ..."
    metadatastring += "\n"
    logfile.write(metadatastring)
    logfile.close()

    outfile = open(outfilename, "w+")
    outfile.write(metadatastring)
    outfile.close()

    xval = xmin
    yval = ymin
    zval = zmin
    while(xval <= xmax and yval <= ymax and zval <= zmax):
        print(xval)

        length = [xval]
        if dimension > 1:
            length.append(yval)
        if dimension > 2:
            length.append(zval)
        
        seconds = runcase(workingdir,
                          length, direction, rcfft, inplace, ntrial,
                          precision, nbatch, devicenum, logfilename)
        #print(seconds)
        with open(outfilename, 'a') as outfile:
            outfile.write(str(dimension))
            outfile.write("\t")
            outfile.write(str(xval))
            outfile.write("\t")
            if(dimension > 1):
                outfile.write(str(yval))
                outfile.write("\t")
            if(dimension > 2):
                outfile.write(str(zval))
                outfile.write("\t")
            outfile.write(str(nbatch))
            outfile.write("\t")
            outfile.write(str(len(seconds)))
            for second in seconds:
                outfile.write("\t")
                outfile.write(str(second))
            outfile.write("\n")
        xval *= radix
        if dimension > 1:
            yval *= radix
        if dimension > 2:
            zval *= radix
        
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
                        
