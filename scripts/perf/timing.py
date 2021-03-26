#!/usr/bin/env python3

import getopt
import subprocess
import sys
import tempfile

from dataclasses import dataclass, field
from pathlib import Path as path
from typing import List

usage = '''A timing script for rocfft

Usage:
\ttiming.py
\t\t-w <string> set test executable path
\t\t-i <string> set test libraries for dloaded libs (appendable)
\t\t-o <string> name of output file (appendable for dload)
\t\t-D <-1,1>   default: -1 (forward).  Direction of transform
\t\t-I          make transform in-place
\t\t-N <int>    number of tests per problem size
\t\t-R          set transform to be real/complex or complex/real
\t\t-d <1,2,3>  default: dimension of transform
\t\t-x <int>    minimum problem size in x direction
\t\t-X <int>    maximum problem size in x direction
\t\t-y <int>    minimum problem size in y direction
\t\t-Y <int>    maximum problem size in Y direction
\t\t-z <int>    minimum problem size in z direction
\t\t-Z <int>    maximum problem size in Z direction
\t\t-f <string> precision: float(default) or double
\t\t-b <int>    batch size
\t\t-g <int>    device number
\t\t-F <file>   filename to read problem sizes from
'''

def resolve(x):
    return path(x).resolve()

def run_rider(prog,
            dload, libdir,
            length, direction, rcfft, inplace, ntrial,
            precision, nbatch, devicenum, logfilename):

    prog = path(prog)

    cmd = [ resolve(prog), "--verbose", 0 ]

    if dload:
        cmd.extend([ "--lib" ] + [ resolve(x) for x in libdir ])

    cmd.extend([ "--device", devicenum ])
    cmd.extend([ "--ntrial", ntrial, "--batchSize", nbatch, "--length" ] + length)

    if precision == "double":
        cmd.append("--double")

    if not inplace:
        cmd.append("-o")

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

    cmd.extend([ "--transformType", ttype, "--itype", itype, "--otype", otype ])

    cmd = [ str(x) for x in cmd ]
    print("Running rider: " + " ".join(cmd))

    fout = tempfile.TemporaryFile(mode="w+")
    proc = subprocess.Popen(cmd, cwd=prog.parent.parent,
                            stdout=fout, stderr=fout)

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
            if line.startswith(searchstr):
                vals.append([])
                # Line ends with "ms", so remove that.
                ms_string = line[len(searchstr): -2]
                for val in ms_string.split():
                    vals[len(vals) - 1].append(1e-3 * float(val))
        print("seconds: ", vals)

    else:
        print("\twell, that didn't work")
        print(rc)
        print(" ".join(cmd))
        return []

    fout.close()

    return vals

# generates a set of lengths starting from the x,y,z min, up to the
# x,y,z max, increasing by specified radix
def radix_size_generator(xmin, ymin, zmin,
                         xmax, ymax, zmax,
                         dimension, radix):
    xval = xmin
    yval = ymin
    zval = zmin
    nbatch = None
    while(xval <= xmax and yval <= ymax and zval <= zmax):
        length = [xval]
        if dimension > 1:
            length.append(yval)
        if dimension > 2:
            length.append(zval)
        yield length, nbatch

        xval *= radix
        if dimension > 1:
            yval *= radix
        if dimension > 2:
            zval *= radix

# generate problem sizes from the specified file.  file should have
# comma-separated dimensions like
#
# 8,16,nbatch=100
# 256,256,64,nbatch=10000
#
# nbatch can be set independently, if not set, use the value in alltime.py
def problem_file_size_generator(problem_file, dimension):
    f = open(problem_file, 'r')
    for line in f:
        if line.startswith('#'):
            continue
        nbatch = None
        lengthBatch = line.replace(' ','').split(',nbatch=')
        if len(lengthBatch) > 1:
            nbatch = int(lengthBatch[1])
        line = lengthBatch[0]
        length = [int(x) for x in line.split(',')]
        if len(length) == dimension:
            yield length, nbatch

@dataclass
class Timer:
    prog: str         = ""
    lib: List[str]    = field(default_factory=list)
    out: List[str]    = field(default_factory=list)
    log: str          = "timing.log"
    device: int       = 0
    ntrial: int       = 10
    direction: int    = -1
    inplace: bool     = False
    real: bool        = False
    precision: str    = "float"
    dimension: int    = 1
    xmin: int         = 2
    xmax: int         = 1024
    ymin: int         = 2
    ymax: int         = 1024
    zmin: int         = 2
    zmax: int         = 1024
    radix: int        = 2
    nbatch: int       = 1
    problem_file: str = None

    def run_cases(self):

        dload = len(self.lib) > 0

        prog = path(self.prog)
        if not prog.is_file():
            print("**** Error: unable to find " + self.prog)
            sys.exit(1)

        metadatastring = "# " + str(self) + "\n"
        metadatastring += "# "
        metadatastring += "dimension"
        metadatastring += "\txlength"
        if(self.dimension > 1):
            metadatastring += "\tylength"
        if(self.dimension > 2):
            metadatastring += "\tzlength"
        metadatastring += "\tnbatch"
        metadatastring += "\tnsample"
        metadatastring += "\tsamples ..."
        metadatastring += "\n"

        # The log file is stored alongside each data output file.
        for out in self.out:
            out = path(out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(metadatastring)

            log = out.with_suffix('.log')
            log.write_text(metadatastring)


        if self.problem_file:
            problem_gen = problem_file_size_generator(self.problem_file, self.dimension)
        else:
            problem_gen = radix_size_generator(self.xmin, self.ymin, self.zmin,
                                              self.xmax, self.ymax, self.zmax,
                                              self.dimension, self.radix)

        for length, nbatch in problem_gen:
            N = self.ntrial
            if nbatch is None:
                nbatch = self.nbatch
            # A possible to-do is to set a fixed data-size and get the adapted nbatch.
            seconds = run_rider(self.prog,
                              dload, self.lib,
                              length, self.direction, self.real, self.inplace, N,
                              self.precision, nbatch, self.device, self.log)
            #print(seconds)
            for idx, vals in enumerate(seconds):
                with open(self.out[idx], 'a') as outfile:
                    outfile.write(str(self.dimension))
                    outfile.write("\t")
                    outfile.write("\t".join([str(val) for val in length]))
                    outfile.write("\t")
                    outfile.write(str(self.nbatch))
                    outfile.write("\t")
                    outfile.write(str(len(seconds[idx])))
                    for second in seconds[idx]:
                        outfile.write("\t")
                        outfile.write(str(second))
                    outfile.write("\n")



def main(argv):

    timer = Timer()

    try:
        opts, args = getopt.getopt(argv,"hb:d:i:D:IN:o:Rw:x:X:y:Y:z:Z:f:r:g:F:")
    except getopt.GetoptError:
        print("error in parsing arguments.")
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h"):
            print(usage)
            exit(0)
        elif opt in ("-w"):
            timer.prog = arg
        elif opt in ("-o"):
            timer.out.append(arg)
        elif opt in ("-i"):
            timer.lib.append(arg)
        elif opt in ("-g"):
            timer.device = int(arg)
        elif opt in ("-N"):
            timer.ntrial = int(arg)
        elif opt in ("-D"):
            if(int(arg) in [-1,1]):
                timer.direction = int(arg)
            else:
                print("invalid direction: " + arg)
                print(usage)
                sys.exit(1)
        elif opt in ("-I"):
            timer.inplace = True
        elif opt in ("-R"):
            timer.real = True
        elif opt in ("-f"):
            if arg not in ["float", "double"]:
                print("precision must be float or double")
                print(usage)
                sys.exit(1)
            timer.precision = arg
        elif opt in ("-d"):
            timer.dimension = int(arg)
            if not timer.dimension in {1,2,3}:
                print("invalid dimension")
                print(usage)
                sys.exit(1)
        elif opt in ("-x"):
            timer.xmin = int(arg)
        elif opt in ("-X"):
            timer.xmax = int(arg)
        elif opt in ("-y"):
            timer.ymin = int(arg)
        elif opt in ("-Y"):
            timer.ymax = int(arg)
        elif opt in ("-z"):
            timer.zmin = int(arg)
        elif opt in ("-Z"):
            timer.zmax = int(arg)
        elif opt in ("-b"):
            timer.nbatch = int(arg)
        elif opt in ("-r"):
            timer.radix = int(arg)
        elif opt in ("-F"):
            timer.problem_file = arg

    timer.run_cases()


if __name__ == "__main__":
    main(sys.argv[1:])
