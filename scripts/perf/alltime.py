#!/usr/bin/env python3
import argparse
from collections import OrderedDict
import sys
from math import *
import subprocess
import os
import re # regexp package
import shutil
import tempfile

import commandrunner as cr

try:
    import numpy as np
except ImportError:
    np = None

def nextpow(val, radix):
    x = 1
    while(x <= val):
        x *= radix
    return x

# A class for generating data for figures.
class RocWavesArgumentSet(cr.ArgumentSetABC):
    def _define_consistent_arguments(self):
        self.consistent_args['nbatch'] = cr.RequiredArgument('-b')
        self.consistent_args['min_size_x'] = cr.RequiredArgument('-x')
        self.consistent_args['max_size_x'] = cr.RequiredArgument('-X')
        self.consistent_args['min_size_y'] = cr.OptionalArgument('-y')
        self.consistent_args['max_size_y'] = cr.OptionalArgument('-Y')
        self.consistent_args['min_size_z'] = cr.OptionalArgument('-z')
        self.consistent_args['max_size_z'] = cr.OptionalArgument('-Z')
        self.consistent_args['radix'] = cr.RequiredArgument('-r')
        self.consistent_args['direction'] = cr.RequiredArgument('-D')
        self.consistent_args['dimension'] = cr.RequiredArgument('-d')
        self.consistent_args['precision'] = cr.RequiredArgument('-f')
        self.consistent_args['inplace'] = cr.OptionalFlagArgument('-I')
        self.consistent_args['fft_type_is_r2c'] = cr.OptionalFlagArgument('-R')

    def _define_variable_arguments(self):
        self.variable_args['nsample'] = cr.RequiredArgument('-N') # TODO: This flag is not currently passed along
        self.variable_args['idir'] = cr.RequiredArgument('-w')
        self.variable_args['output_file'] = cr.RequiredArgument('-o')

    def __init__(self, combine_executables, labelsuffix,
                 dimension, minsize, maxsize, nbatch, radix, ratio, ffttype,
                 direction, inplace):
        cr.ArgumentSetABC.__init__(
                self,
                combine_executables = combine_executables,
                nbatch = nbatch,
                min_size_x = minsize,
                max_size_x = maxsize,
                radix = radix,
                direction = direction,
                dimension = dimension,
                precision = "double",
                inplace = inplace,
                fft_type_is_r2c = (ffttype == "r2c"),
                )
        if dimension > 1:
            self.set('min_size_y', minsize * ratio[0])
            self.set('max_size_y', maxsize * ratio[0])

        if dimension > 2:
            self.set('min_size_z', minsize * ratio[1])
            self.set('max_size_z', maxsize * ratio[1])

        self.labelsuffix = labelsuffix
        self.minsize = minsize
        self.maxsize = maxsize
        self.ratio = ratio
        self.ffttype = ffttype

    def get_output_basename(self):
        outfile = ""
        outfile += "radix" + str(self._radix)
        outfile += "_dim" + str(self._dimension)
        outfile += "_" + self._precision
        outfile += "_n" + str(self._nbatch)
        if self._direction == 1:
            outfile += "_inv"
        if self._dimension > 1:
            outfile += "_ratio" + "_" + str(self.ratio[0])
        if self._dimension > 2:
            outfile += "_" + str(self.ratio[1])
        outfile += "_" + self.ffttype
        if self._inplace:
            outfile += "_inplace"
        else:
            outfile += "_outofplace"
        outfile += ".dat"
        return outfile

    def get_caption(self, similar_keys):
        caption = []
        if 'dimension' in similar_keys:
            caption.append('{}D'.format(self._dimension))

        if 'inplace' in similar_keys:
            caption.append('in-place' if self._inplace else 'out-of-place')

        if 'fft_type_is_r2c' in similar_keys and 'direction' in similar_keys:
            direction_str = 'complex-to-complex'
            if self._fft_type_is_r2c and self._direction == 1:
                direction_str = 'complex-to-real'
            if self._fft_type_is_r2c and self._direction == 0:
                direction_str = 'real-to-complex'
            caption.append(direction_str + ' transforms')

        if self._dimension > 1:
            if 'max_size_y' in similar_keys:
                ratio_str = 'aspect ratio N:{}N'.format('' if self.ratio[0] == 1 else self.ratio[0])
                if self._dimension > 2:
                    ratio_str += ':{}N'.format('' if self.ratio[1] == 1 else self.ratio[1])
                caption.append(ratio_str)

        if 'radix' in similar_keys:
            caption.append('radix {}'.format(self._radix))

        if 'nbatch' in similar_keys:
            caption.append('batch size {}'.format(self._nbatch))
        return ', '.join(caption[:-1]) + ' and {}.'.format(caption[-1])

    def collect_timing(self, run_configuration):
        output_filename = self.get_output_file(run_configuration)
        rv = {}
        print('Processing {}'.format(output_filename))
        if os.path.exists(output_filename) and np is not None:
            # The output format is not consistent enough to justify using an out of the box reader.
            # raw_tsv = np.genfromtxt(output_filename,
            #                         delimiter='\t',
            #                         skip_header = 1,
            #                         names = None, # would only work if there is a label for each sub-sample
            #                         invalid_raise=False,
            #                         )
            # # For each xlength, store all of the samples
            # if raw_tsv.ndim == 1:
            #     raw_tsv = [raw_tsv] # For a single line in the CSV, genfromtxt unfortunately returns a 1D array
            # for tsv_line in raw_tsv:
            #     if len(tsv_line) > 0:
            #         dim = int(tsv_line[0])
            #         if len(tsv_line) > dim+3:
            #             xlength = int(tsv_line[1])
            #             samples = tsv_line[dim+3:]
            #             rv[xlength] = samples
            #         else:
            #             print('WARNING: Could not parse {}'.format(output_filename))
            with open(output_filename, 'r') as raw_tsv:
                for line in raw_tsv.readlines():
                    # remove comment by splittling on `#` and taking the first segment
                    stripped_line = line.split('#')[0].strip()
                    if stripped_line:
                        split_line = stripped_line.split('\t')
                        # Parse dimension
                        dimension = int(split_line[0]) if len(split_line) > 0 else 0
                        read_idx = 1
                        volume_size = 1
                        # Parse volume size
                        for i in range(dimension):
                            if len(split_line) > read_idx:
                                volume_size *= int(split_line[read_idx])
                                read_idx += 1
                        # Parse num batches
                        num_batches = int(split_line[read_idx]) if len(split_line) > read_idx else 0
                        read_idx += 1
                        # Parse num samples
                        num_samples = int(split_line[read_idx]) if len(split_line) > read_idx else 0
                        read_idx += 1
                        # Check consistency
                        if num_samples > 0:
                            samples = [float(x) for x in split_line[read_idx:]]
                            if len(samples) != num_samples:
                                print('WARNING: Inconsistency when parsing TSV')
                            rv[volume_size] = samples

        else:
            print('{} does not exist'.format(output_filename))
        return rv

class StaticRiderArgumentSet(RocWavesArgumentSet):
    def _define_variable_arguments(self):
        self.variable_args['nsample'] = cr.RequiredArgument('-N') # TODO: This flag is not currently passed along
        self.variable_args['idir'] = cr.RequiredArgument('-w')
        self.variable_args['output_file'] = cr.RequiredArgument('-o')

    def __init__(self, *args, **kwargs):
        RocWavesArgumentSet.__init__(self, False, *args, **kwargs)

    def get_full_command(self, run_configuration):
        timingscript = './timing.py'
        if not os.path.exists(timingscript):
            timingscript = os.path.join(os.path.dirname(os.path.realpath(__file__)), timingscript)
        else:
            timingscript = os.path.abspath(timingscript)
        if not os.path.exists(timingscript):
            raise RuntimeError("Unable to find {}!".format(timingscript))

        self.set('nsample', run_configuration.num_runs)
        self.set('idir', os.path.abspath(run_configuration.executable_directory))
        self.set('output_file', os.path.abspath(self.get_output_file(run_configuration)))

        return [timingscript] + self.get_args()

class DynaRiderArgumentSet(RocWavesArgumentSet):
    def _define_variable_arguments(self):
        self.variable_args['nsample'] = cr.RequiredArgument('-N') # TODO: This flag is not currently passed along
        self.variable_args['dload_binary'] = cr.RequiredArgument('-w')
        self.variable_args['input_libraries'] = cr.RepeatedArgument('-i')
        self.variable_args['output_files'] = cr.RepeatedArgument('-o')

    def __init__(self, *args, **kwargs):
        RocWavesArgumentSet.__init__(self, True, *args, **kwargs)

    def get_interleaved_command(self, run_configurations):
        timingscript = './timing.py'
        if not os.path.exists(timingscript):
            timingscript = os.path.join(os.path.dirname(os.path.realpath(__file__)), timingscript)
        else:
            timingscript = os.path.abspath(timingscript)
        if not os.path.exists(timingscript):
            raise RuntimeError("Unable to find {}!".format(timingscript))

        self.set('nsample', run_configurations[0].num_runs)
        self.set('dload_binary', os.path.abspath(self.user_args.dload_binary))
        self.set('input_libraries', [os.path.abspath(run_configuration.executable_directory) for run_configuration in run_configurations])
        self.set('output_files', [os.path.abspath(self.get_output_file(run_configuration)) for run_configuration in run_configurations])

        return [timingscript] + self.get_args()

class RocFftRunConfiguration(cr.RunConfiguration):
    def __init__(self, user_args, *args, **kwargs):
        cr.RunConfiguration.__init__(self, user_args, *args, **kwargs)
        self.num_runs = user_args.num_runs


# Figure class, which contains runs and provides commands to generate figures.
class figure(cr.Comparison):
    def __init__(self, filename):
        cr.Comparison.__init__(self, filename=filename)
        self.runs = self.argument_sets

    def inputfiles(self, outdirlist):
        files = []
        for run in self.runs:
            for outdir in outdirlist:
                files.append(os.path.join(outdir, run.get_output_basename()))
        return files

    def labels(self, labellist):
        labels = []
        for run in self.runs:
            for label in labellist:
                labels.append(label + " " + run.labelsuffix)
        return labels

    def getfilename(self, outdir, docformat):
        outfigure = self.get_name()
        outfigure += ".pdf"
        # if docformat == "pdf":
        #     outfigure += ".pdf"
        # if docformat == "docx":
        #     outfigure += ".png"
        return os.path.join(outdir, outfigure)

    def plot(self, run_configurations, axes):
        label_map = OrderedDict()
        xlengths_map = OrderedDict()
        samples_map = OrderedDict()
        # Combine equivalent run configurations
        for run_configuration in run_configurations:
            for run in self.runs:
                key = run_configuration.get_id() + run.get_hash()
                xlengths = xlengths_map[key] if key in xlengths_map else []
                samples = samples_map[key] if key in samples_map else []
                for xlength, subsamples in run.collect_timing(run_configuration).items():
                    for sample in subsamples:
                        xlengths.append(xlength)
                        samples.append(sample)
                label_map[key] = run_configuration.label + ' ' + run.labelsuffix
                xlengths_map[key] = xlengths
                samples_map[key] = samples
        for key in label_map:
            axes.loglog(xlengths_map[key], samples_map[key], '.',
                        label = label_map[key],
                        markersize = 3,
                        )
        axes.set_xlabel('x-length (integer)')
        axes.set_ylabel('Time (s)')

    def asycmd(self, run_configurations):
        docdir = self.user_args.documentation_directory
        docformat = self.user_args.doc_format
        datatype = self.user_args.data_type
        ncompare =  len(run_configurations) if self.user_args.speedup else 0
        secondtype = self.user_args.second_type
        just1dc2crad2 = (self.user_args.run_type == ["efficiency"])
        outdirlist = [run_configuration.output_directory for run_configuration in run_configurations]
        labellist = [run_configuration.label for run_configuration in run_configurations]
        asycmd = ["asy"]

        asycmd.append("-f")
        asycmd.append("pdf")

        asycmd.append("datagraphs.asy")


        asycmd.append("-u")
        inputfiles = self.inputfiles(outdirlist)
        asycmd.append('filenames="' + ",".join(inputfiles) + '"')

        asycmd.append("-u")
        asycmd.append('legendlist="' + ",".join(self.labels(labellist)) + '"')

        asycmd.append("-u")
        asycmd.append('speedup=' + str(ncompare))

        if just1dc2crad2 :
            asycmd.append("-u")
            asycmd.append('just1dc2crad2=true')

        if secondtype == "gflops":
            asycmd.append("-u")
            asycmd.append('secondarygflops=true')

        if datatype == "gflops":
            asycmd.append("-u")
            asycmd.append('primaryaxis="gflops"')

        if datatype == "roofline":
            asycmd.append("-u")
            asycmd.append('primaryaxis="roofline"')
            # roofline on multiple devices doesn't really make sense; just use the first device
            with open(os.path.join(outdirlist[0], "gpuid.txt"), "r") as f:
                gpuid = f.read()
                asycmd.append("-u")
                asycmd.append('gpuid="' + gpuid.strip() + '"')

        if len(self.runs) > 0:
            asycmd.append("-u")
            asycmd.append('batchsize=' + str(self.runs[0]._nbatch))
            asycmd.append("-u")
            asycmd.append('problemdim=' + str(self.runs[0]._dimension))
            asycmd.append("-u")
            val = 1
            for rat in self.runs[0].ratio:
                val *= rat
            asycmd.append('problemratio=' + str(val))
            asycmd.append("-u")
            if self.runs[0].ffttype == "r2c":
                asycmd.append("realcomplex=true")
            else:
                asycmd.append("realcomplex=false")

        asyfilename = self.getfilename(docdir, docformat)
        asycmd.append("-o")
        asycmd.append(asyfilename)
        return asycmd, asyfilename

    def custom_plot(self, run_configurations, is_make_plot):
        asycmd, asyfilename = self.asycmd(run_configurations)
        if is_make_plot:
            print(" ".join(asycmd))

            fout = tempfile.TemporaryFile(mode="w+")
            ferr = tempfile.TemporaryFile(mode="w+")
            asyproc = subprocess.Popen(asycmd,
                                    stdout=fout, stderr=ferr, env=os.environ.copy(), cwd = os.sys.path[0])
            asyproc.wait()
            asyrc = asyproc.returncode
            if asyrc != 0:
                print("****asy fail****")
                fout.seek(0)
                cout = fout.read()
                print(cout)
                ferr.seek(0)
                cerr = ferr.read()
                print(cerr)
        nsample = run_configurations[0].num_runs
        plot_caption = self.get_caption()
        plot_caption += " Each data point represents the median of " + str(nsample) + " values, with error bars showing the 95% confidence interval for the median.  All transforms are double-precision."
        return asyfilename, plot_caption

# Function for generating figures for benchmark output
def benchfigs(rundims, shortrun, dynarider):
    rundata = DynaRiderArgumentSet if dynarider else StaticRiderArgumentSet
    figs = []
    # FFT directions
    forwards = -1
    backwards = 1


    if 1 in rundims:
        dimension = 1

        nbatch = 1

        min1d = 256 if shortrun else 1024
        max1d = 4000 if shortrun else 536870912

        for inplace in [True, False]:
            fig = figure(filename="1d_c2c" + ("inplace" if inplace else "outofplace"))
            for radix in [2, 3]:
                fig.runs.append( rundata("radix " + str(radix) ,
                                         dimension, nextpow(min1d, radix), max1d, nbatch,
                                         radix, [], "c2c", forwards, inplace) )
            figs.append(fig)

        for inplace in [True, False]:
            fig = figure(filename="1d_r2c" + ("inplace" if inplace else "outofplace"))
            for radix in [2, 3]:
                fig.runs.append( rundata("radix " + str(radix) ,
                                         dimension, nextpow(min1d, radix), max1d, nbatch,
                                         radix, [], "r2c", forwards, inplace) )
            figs.append(fig)


        for inplace in [True, False]:
            fig = figure(filename="1d_c2r" + ("inplace" if inplace else "outofplace"))
            for radix in [2, 3]:
                fig.runs.append( rundata("radix " + str(radix) ,
                                         dimension, nextpow(min1d, radix), max1d, nbatch,
                                         radix, [], "r2c", backwards, inplace) )
            figs.append(fig)


    if 2 in rundims:
        dimension = 2

        nbatch = 1
        min2d = 64 if shortrun else 128
        max2d = 8192 if shortrun else 32768

        for inplace in [True, False]:
            fig = figure(filename="2d_c2c" + ("inplace" if inplace else "outofplace"))
            for radix in [2, 3]:
                fig.runs.append( rundata("radix " + str(radix), dimension,
                                         nextpow(min2d, radix), max2d, nbatch, radix, [1], "c2c",
                                         forwards, inplace) )
            figs.append(fig)

        for inplace in [True, False]:
            fig = figure(filename="2d_r2c" + ("inplace" if inplace else "outofplace"))
            for radix in [2, 3]:
                fig.runs.append( rundata("radix " + str(radix), dimension,
                                         nextpow(min2d, radix), max2d, nbatch, radix, [1], "r2c",
                                         forwards, inplace) )
            figs.append(fig)

        for inplace in [True, False]:
            fig = figure(filename="2d_c2r" + ("inplace" if inplace else "outofplace"))
            for radix in [2, 3]:
                fig.runs.append( rundata("radix " + str(radix), dimension,
                                         nextpow(min2d, radix), max2d, nbatch, radix, [1], "r2c",
                                         backwards, inplace) )
            figs.append(fig)

    if 3 in rundims:
        dimension = 3
        min3d = 16
        max3d = 128 if shortrun else 1024
        nbatch = 1

        for inplace in [True]:
            fig = figure(filename="3d_c2c" + ("inplace" if inplace else "outofplace"))
            for radix in [2, 3, 5]:
                fig.runs.append( rundata("radix " + str(radix), dimension,
                                        nextpow(min3d, radix), max3d, nbatch, radix, [1,1],
                                        "c2c",
                                        forwards, inplace) )
            figs.append(fig)

        for inplace in [True, False]:
            fig = figure(filename="3d_r2c" + ("inplace" if inplace else "outofplace"))
            for radix in [2, 3]:
                fig.runs.append( rundata("radix " + str(radix), dimension,
                                         nextpow(min3d, radix), max3d, nbatch, radix, [1,1], "r2c",
                                         forwards, inplace) )
            figs.append(fig)

        for inplace in [True, False]:
            fig = figure(filename="3d_c2r" + ("inplace" if inplace else "outofplace"))
            for radix in [2, 3]:
                fig.runs.append( rundata("radix " + str(radix), dimension,
                                         nextpow(min3d, radix), max3d, nbatch, radix, [1,1], "r2c",
                                         backwards, inplace) )
            figs.append(fig)

    return figs

def efficiencyfigs(rundims, shortrun, dynarider):
    rundata = DynaRiderArgumentSet if dynarider else StaticRiderArgumentSet
    figs = []

    # FFT directions
    forwards = -1
    backwards = 1

    inplace = True
    dimension = 1

    radix = 2

    min1d = 1024
    max1d = 1048576 if shortrun else 268435456 #pow(2,28) gives a floating type :(
    nbatch = 1
    while max1d > min1d:
        fig = figure(filename="1d_c2c_batch" + str(nbatch) + "_radix" + str(radix))

        fig.runs.append( rundata("",
                                 dimension, nextpow(min1d, radix), max1d, nbatch,
                                 radix, [], "c2c", forwards, inplace) )
        figs.append(fig)
        nbatch *= 2
        max1d //= 2
        min1d //= 2
        min1d = max(min1d, 2^5)
    return figs

# Function for generating figures for a performance report
def reportfigs(rundims, shortrun, dynarider):
    rundata = DynaRiderArgumentSet if dynarider else StaticRiderArgumentSet
    figs = []

    # FFT directions
    forwards = -1
    backwards = 1

    inplace = True

    if 1 in rundims:
        dimension = 1

        for min1d, max1d, nbatch in [[1024,536870912,1], [8,32768,100000]]:

            for radix in [2, 3, 5, 7]:
                fig = figure(filename="1d_c2c" \
                             + "_radix" + str(radix) \
                             + "_batch" + str(nbatch))
                fig.runs.append( rundata("",
                                         dimension, nextpow(min1d, radix),
                                         max1d, nbatch,
                                         radix, [], "c2c", forwards,
                                         inplace) )
                figs.append(fig)

            for radix in [2, 3, 5, 7]:
                fig = figure(filename="1d_r2c"\
                             + "_radix" + str(radix) \
                             + "_batch" + str(nbatch))
                fig.runs.append( rundata("",
                                         dimension, nextpow(min1d, radix),
                                         max1d, nbatch,
                                         radix, [], "r2c", forwards,
                                         inplace) )
                figs.append(fig)

            for radix in [2, 3, 5, 7]:
                fig = figure(filename="1d_c2r" \
                             + "_radix" + str(radix) \
                             + "_batch" + str(nbatch))
                fig.runs.append( rundata("radix "+str(radix),
                                         dimension, nextpow(min1d, radix),
                                         max1d, nbatch,
                                         radix, [], "r2c", backwards,
                                         inplace) )
                figs.append(fig)

    if 2 in rundims:
        dimension = 2

        for min2d, max2d, nbatch in [[128,32768,1], [64,8192,100]]:

            for radix in [2, 3, 5]:
                fig = figure(filename="2d_c2c" \
                             + "_radix" + str(radix) \
                             + "_batch" + str(nbatch))
                fig.runs.append( rundata("radix "+ str(radix),
                                         dimension,
                                         nextpow(min2d, radix), max2d,
                                         nbatch,
                                         radix, [1], "c2c",
                                         forwards, inplace) )
                figs.append(fig)

            for radix in [2, 3, 5]:
                fig = figure(filename="2d_r2c" \
                             + "_radix" + str(radix) \
                             + "_batch" + str(nbatch))

                fig.runs.append( rundata("radix " + str(radix),
                                         dimension,
                                         nextpow(min2d, radix), max2d,
                                         nbatch,
                                         radix, [1], "r2c",
                                         forwards, inplace) )
                figs.append(fig)

            for radix in [2, 3, 5]:
                fig = figure(filename="2d_c2r" \
                             + "_radix" + str(radix) \
                             + "_batch" + str(nbatch))
                fig.runs.append( rundata("radix " + str(radix),
                                         dimension,
                                         nextpow(min2d, radix), max2d,
                                         nbatch,
                                         radix, [1], "r2c",
                                         backwards, inplace) )
                figs.append(fig)


            for radix in [2]:
                fig = figure(filename="2d_c2c_r2" \
                             + "_radix" + str(radix) \
                             + "_batch" + str(nbatch))
                fig.runs.append( rundata("radix 2",
                                         dimension, min2d, max2d, nbatch, 2,
                                         [2], "c2c",
                                         forwards, inplace) )
                figs.append(fig)

            for radix in [2]:
                fig = figure(filename="2d_r2c_r2" \
                             + "_radix" + str(radix) \
                             + "_batch" + str(nbatch))
                fig.runs.append( rundata("radix 2",
                                         dimension, min2d, max2d, nbatch, 2,
                                         [2], "r2c",
                                         forwards, inplace) )
                figs.append(fig)

    if 3 in rundims:
        dimension = 3

        for min3d, max3d, nbatch in [[16,128,1],[4,64,100]]:

            for radix in [2, 3, 5]:
                fig = figure(filename="3d_c2c" \
                             + "_radix" + str(radix) \
                             + "_batch" + str(nbatch))
                fig.runs.append( rundata("radix " + str(radix),
                                         dimension,
                                         nextpow(min3d, radix), max3d,
                                         nbatch,
                                         radix, [1,1], "c2c",
                                         forwards, inplace) )
                figs.append(fig)

            for radix in [2, 3]:
                fig = figure(filename="3d_r2c" \
                             + "_radix" + str(radix) \
                             + "_batch" + str(nbatch))

                fig.runs.append( rundata("radix " + str(radix),
                                         dimension,
                                         nextpow(min3d, radix), max3d,
                                         nbatch,
                                         radix, [1,1], "r2c",
                                         forwards, inplace) )
                figs.append(fig)


            fig = figure(filename="3d_c2r" \
                         + "_radix" + str(radix) \
                         + "_batch" + str(nbatch))
            for radix in [2]:
                fig.runs.append( rundata("radix " + str(radix),
                                         dimension,
                                         nextpow(min3d, radix), max3d,
                                         nbatch,
                                         radix, [1,1], "r2c",
                                         backwards, inplace) )
            figs.append(fig)

            fig = figure(filename="3d_c2c_aspect" \
                         + "_radix" + str(radix) \
                         + "_batch" + str(nbatch))
            fig.runs.append( rundata("radix 2",
                                     dimension, min3d, max3d, nbatch, 2,
                                     [1,16], "c2c",
                                     forwards, inplace) )
            figs.append(fig)

            fig = figure(filename="3d_r2c_aspect" \
                         + "_radix" + str(radix) \
                         + "_batch" + str(nbatch))
            fig.runs.append( rundata("radix 2",
                                     dimension, min3d, max3d, nbatch, 2,
                                     [1,16], "r2c",
                                     forwards, inplace) )
            figs.append(fig)

    return figs


def main(argv):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value or next flag expected.')
    # WARNING: The approach to using booleans is not compatible with purely positional arguments
    #    because `--bool-option <positional arg>` would attempt to convert the positional argument
    #    to a boolean. Otherwise, it is more intuitive and flexible than enable/disable flags.

    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--is-short-run',
                        type=str2bool, nargs='?', const=True, default=False,
                        help='Run a shorter test.')
    parser.add_argument('-D', '--dimension', default=[1,2,3], type=int, nargs='+',
                        help='Space separated list of dimensions to run.')
    parser.add_argument('-N', '--num-runs', default=10, type=int,
                        help='Number of times to run each test.')
    parser.add_argument('-b', '--dload-binary', default=None,
                        help='Optional binary for combining different dynamic libraries into a single executable using dload.')
    parser.add_argument('-S', '--speedup',
                        type=str2bool, nargs='?', const=True, default=True,
                        help='Plot speedup with respect to the first run.')
    parser.add_argument('-t', '--data-type', default='time',
                        choices = ['time', 'gflops', 'roofline'],
                        help='Primary axis dataype.')
    parser.add_argument('-y', '--second-type', default='none',
                        choices = ['none', 'gflops'],
                        help='Secondary axis dataype.')
    parser.add_argument('-R', '--run-type', default=['benchmark'], nargs='+',
                        choices=['report', 'benchmark', 'efficiency'],
                        help='Secondary axis dataype.')
    parser.add_argument('-f', '--doc-format', default='pdf',
                        choices=['pdf', 'docx'],
                        help='Documentation output format.')
    user_args = cr.parse_input_arguments(parser)

    command_runner = cr.CommandRunner(user_args, RocFftRunConfiguration)

    if user_args.num_repititions > 1 and command_runner.is_make_document():
        print("WARNING - document generation does not support multiple coarse grained runs")
        import time
        time.sleep(5)

    rundims = user_args.dimension
    print("rundims:")
    print(rundims)

    indirlist = command_runner.executable_directories
    outdirlist = command_runner.output_directories
    labellist = command_runner.labels

    # if command_runner.is_run_tool():
    #     for indir in indirlist:
    #         if not binaryisok(indir, "rocfft-rider"):
    #             print("unable to find " + "rocfft-rider" + " in " + indir)
    #             print("please specify with -i")
    #             sys.exit(1)

    if user_args.is_short_run:
        print("short run")
    print("output format: " + user_args.doc_format)
    print("device number: " + str(user_args.device_num))

    command_runner.setup_system()

    figs = []

    dynarider = user_args.dload_binary is not None
    if "benchmark" in user_args.run_type:
        figs += benchfigs(rundims, user_args.is_short_run, dynarider)
    if "report" in user_args.run_type:
        figs += reportfigs(rundims, user_args.is_short_run, dynarider)
    if "efficiency" in user_args.run_type:
        figs += efficiencyfigs(rundims, user_args.is_short_run, dynarider)
    just1dc2crad2 = user_args.run_type == "efficiency"

    for idx, fig in enumerate(figs):
        for idx2, fig2 in enumerate(figs):
            if idx != idx2 and fig.get_name() == fig2.get_name():
                print("figures have the same name!")
                print(fig.get_name())
                print(fig.runs)
                print(fig2.get_name())
                print(fig2.runs)
                sys.exit(1)

    command_runner.add_comparisons(figs)

    command_runner.execute()

    command_runner.show_plots()
    command_runner.output_summary()

def binaryisok(dirname, progname):
    prog = os.path.join(dirname, progname)
    return os.path.isfile(prog)

gflopstext = '''\
GFLOP/s are computed based on the Cooley--Tukey operation count \
for a radix-2 transform, and half that for in the case of \
real-complex transforms.  The rocFFT operation count may differ from \
this value: GFLOP/s is provided for the sake of comparison only.'''

# Confert a PDF to an EMF using pdf2svg and inkscape.
def pdf2emf(pdfname):
    svgname = pdfname.replace(".pdf",".svg")
    cmd_pdf2svg = ["pdf2svg", pdfname, svgname]
    proc = subprocess.Popen(cmd_pdf2svg, env=os.environ.copy())
    proc.wait()
    if proc.returncode != 0:
        print("pdf2svg failed!")
        sys.exit(1)

    emfname = pdfname.replace(".pdf",".emf")
    cmd_svg2emf = ["inkscape", svgname, "-M", emfname]
    proc = subprocess.Popen(cmd_svg2emf, env=os.environ.copy())
    proc.wait()
    if proc.returncode != 0:
        print("svg2emf failed!")
        sys.exit(1)

    return emfname

# Function for generating a docx using emf files and the docx package.
def makedocx(figs, outdir, nsample, secondtype):
    import docx

    document = docx.Document()

    document.add_heading('rocFFT benchmarks', 0)

    document.add_paragraph("Each data point represents the median of " + str(nsample) + " values, with error bars showing the 95% confidence interval for the median.  Transforms are double-precision, forward, and in-place.")

    if secondtype == "gflops":
        document.add_paragraph(gflopstext)

    specfilename = os.path.join(outdir, "specs.txt")
    if os.path.isfile(specfilename):
        with open(specfilename, "r") as f:
            specs = f.read()
        for line in specs.split("\n"):
            document.add_paragraph(line)

    for fig in figs:
        print(fig.getfilename(outdir, "docx"))
        print(fig.get_caption())
        emfname = pdf2emf(fig.getfilename(outdir, "docx"))
        document.add_picture(emfname, width=docx.shared.Inches(6))
        document.add_paragraph(fig.get_caption())

    document.save(os.path.join(outdir,'figs.docx'))

if __name__ == "__main__":
    main(sys.argv[1:])

