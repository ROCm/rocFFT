#!/usr/bin/env python3
"""rocFFT kernel generator.

It accept two sub-commands:
1. list - lists files that will be generated
2. generate - generate them!

"""

import argparse
import collections
import functools
import itertools
import subprocess
import sys
import os

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from operator import mul

from generator import (ArgumentList, BaseNode, Call, CommentBlock, Function, Include,
                       LineBreak, Map, StatementList, Variable, name_args, write,
                       clang_format_file)


from collections import namedtuple

LaunchParams = namedtuple('LaunchParams', ['transforms_per_block',
                                           'workgroup_size',
                                           'threads_per_transform',
                                           'half_lds'])

#
# CMake helpers
#

def scjoin(xs):
    """Join 'xs' with semi-colons."""
    return ';'.join(str(x) for x in xs)


def scprint(xs):
    """Print 'xs', joined by semi-colons, on a single line.  CMake friendly."""
    print(scjoin(xs), end='', flush=True)


def cjoin(xs):
    """Join 'xs' with commas."""
    return ','.join(str(x) for x in xs)


#
# Helpers
#

def flatten(lst):
    """Flatten a list of lists to a list."""
    return sum(lst, [])


def unique(kernels):
    """Merge kernel lists without duplicated meta.length; ignore later ones."""
    r, s = list(), set()
    for kernel in kernels:
        if isinstance(kernel.length, list):
            key = tuple(kernel.length) + (kernel.scheme,)
        else:
            key = (kernel.length, kernel.scheme)
        if key not in s:
            s.add(key)
            r.append(kernel)
    return r

#
# Prototype generators
#

@name_args(['function'])
class FFTKernel(BaseNode):
    def __str__(self):
        f = 'FFTKernel('
        if self.function.meta.runtime_compile:
            f += 'nullptr'
        else:
            f += str(self.function.address())
        use_3steps_large_twd = getattr(self.function.meta, 'use_3steps_large_twd', None)
        if use_3steps_large_twd is not None:
            f += ', ' + str(use_3steps_large_twd[self.function.meta.precision])
        else:
            f += ', false'
        factors = getattr(self.function.meta, 'factors', None)
        if factors is not None:
            f += ', {' + cjoin(factors) + '}'
        transforms_per_block = getattr(self.function.meta, 'transforms_per_block', None)
        if transforms_per_block is not None:
            f += ', ' + str(transforms_per_block)
        workgroup_size = getattr(self.function.meta, 'workgroup_size', None)
        if workgroup_size is not None:
            f += ', ' + str(workgroup_size)
        f += ', {' + ','.join([str(s) for s in self.function.meta.threads_per_transform]) + '}'
        half_lds = None
        if hasattr(self.function.meta, 'params'):
            half_lds = getattr(self.function.meta.params, 'half_lds', None)
        if half_lds is not None:
            f += ', ' + str(half_lds).lower()
        f += ')'
        return f


def generate_cpu_function_pool(functions):
    """Generate function to populate the kernel function pool."""

    function_map = Map('function_map')
    precisions = { 'sp': 'rocfft_precision_single',
                   'dp': 'rocfft_precision_double' }

    populate = StatementList()
    for f in functions:
        length, precision, scheme, transpose = f.meta.length, f.meta.precision, f.meta.scheme, f.meta.transpose
        if isinstance(length, (int, str)):
            length = [length, 0]
        key = Call(name='std::make_tuple',
                   arguments=ArgumentList('std::array<size_t, 2>({' + cjoin(length) + '})',
                                          precisions[precision],
                                          scheme,
                                          transpose or 'NONE')).inline()
        populate += function_map.assert_emplace(key, FFTKernel(f))

    return StatementList(
        Include('<iostream>'),
        Include('"../include/function_pool.h"'),
        StatementList(*[f.prototype() for f in functions]),
        Function(name='function_pool::function_pool',
                 value=False,
                 arguments=ArgumentList(),
                 body=populate))


def list_generated_kernels(kernels):
    """Return list of kernel filenames."""
    return [kernel_file_name(x) for x in kernels if not x.runtime_compile]


#
# Main!
#

def kernel_file_name(ns):
    """Given kernel info namespace, return reasonable file name."""

    assert hasattr(ns, 'length')
    length = ns.length

    if isinstance(length, (tuple, list)):
        length = 'x'.join(str(x) for x in length)

    postfix = ''
    if ns.scheme == 'CS_KERNEL_STOCKHAM_BLOCK_CC':
        postfix = '_sbcc'
    elif ns.scheme == 'CS_KERNEL_STOCKHAM_BLOCK_RC':
        postfix = '_sbrc'
    elif ns.scheme == 'CS_KERNEL_STOCKHAM_BLOCK_CR':
        postfix = '_sbcr'

    return f'rocfft_len{length}{postfix}.cpp'


def list_small_kernels():
    """Return list of small kernels to generate."""

    kernels1d = [
        NS(length=   1, workgroup_size= 64, threads_per_transform=  1, factors=(1,)),
        NS(length=   2, workgroup_size= 64, threads_per_transform=  1, factors=(2,)),
        NS(length=   3, workgroup_size= 64, threads_per_transform=  1, factors=(3,)),
        NS(length=   4, workgroup_size=128, threads_per_transform=  1, factors=(4,)),
        NS(length=   5, workgroup_size=128, threads_per_transform=  1, factors=(5,)),
        NS(length=   6, workgroup_size=128, threads_per_transform=  1, factors=(6,)),
        NS(length=   7, workgroup_size= 64, threads_per_transform=  1, factors=(7,)),
        NS(length=   8, workgroup_size= 64, threads_per_transform=  4, factors=(4, 2)),
        NS(length=   9, workgroup_size= 64, threads_per_transform=  3, factors=(3, 3)),
        NS(length=  10, workgroup_size= 64, threads_per_transform=  1, factors=(10,)),
        NS(length=  11, workgroup_size=128, threads_per_transform=  1, factors=(11,)),
        NS(length=  12, workgroup_size=128, threads_per_transform=  6, factors=(6, 2)),
        NS(length=  13, workgroup_size= 64, threads_per_transform=  1, factors=(13,)),
        NS(length=  14, workgroup_size=128, threads_per_transform=  7, factors=(7, 2)),
        NS(length=  15, workgroup_size=128, threads_per_transform=  5, factors=(3, 5)),
        NS(length=  16, workgroup_size= 64, threads_per_transform=  4, factors=(4, 4)),
        NS(length=  17, workgroup_size=256, threads_per_transform=  1, factors=(17,)),
        NS(length=  18, workgroup_size= 64, threads_per_transform=  6, factors=(3, 6)),
        NS(length=  20, workgroup_size=256, threads_per_transform= 10, factors=(5, 4)),
        NS(length=  21, workgroup_size=128, threads_per_transform=  7, factors=(3, 7)),
        NS(length=  22, workgroup_size= 64, threads_per_transform=  2, factors=(11, 2)),
        NS(length=  24, workgroup_size=256, threads_per_transform=  8, factors=(8, 3)),
        NS(length=  25, workgroup_size=256, threads_per_transform=  5, factors=(5, 5)),
        NS(length=  26, workgroup_size= 64, threads_per_transform=  2, factors=(13, 2)),
        NS(length=  27, workgroup_size=256, threads_per_transform=  9, factors=(3, 3, 3)),
        NS(length=  28, workgroup_size= 64, threads_per_transform=  4, factors=(7, 4)),
        NS(length=  30, workgroup_size=128, threads_per_transform= 10, factors=(10, 3)),
        NS(length=  32, workgroup_size= 64, threads_per_transform= 16, factors=(16, 2)),
        NS(length=  36, workgroup_size= 64, threads_per_transform=  6, factors=(6, 6)),
        NS(length=  40, workgroup_size=128, threads_per_transform= 10, factors=(10, 4)),
        NS(length=  42, workgroup_size=256, threads_per_transform=  7, factors=(7, 6)),
        NS(length=  44, workgroup_size= 64, threads_per_transform=  4, factors=(11, 4)),
        NS(length=  45, workgroup_size=128, threads_per_transform= 15, factors=(5, 3, 3)),
        NS(length=  48, workgroup_size= 64, threads_per_transform= 16, factors=(4, 3, 4)),
        NS(length=  49, workgroup_size= 64, threads_per_transform=  7, factors=(7, 7)),
        NS(length=  50, workgroup_size=256, threads_per_transform= 10, factors=(10, 5)),
        NS(length=  52, workgroup_size= 64, threads_per_transform=  4, factors=(13, 4)),
        NS(length=  54, workgroup_size=256, threads_per_transform= 18, factors=(6, 3, 3)),
        NS(length=  56, workgroup_size=128, threads_per_transform=  8, factors=(7, 8)),
        NS(length=  60, workgroup_size= 64, threads_per_transform= 10, factors=(6, 10)),
        NS(length=  64, workgroup_size= 64, threads_per_transform= 16, factors=(4, 4, 4)),
        NS(length=  72, workgroup_size= 64, threads_per_transform=  9, factors=(8, 3, 3)),
        NS(length=  75, workgroup_size=256, threads_per_transform= 25, factors=(5, 5, 3)),
        NS(length=  80, workgroup_size= 64, threads_per_transform= 10, factors=(5, 2, 8)),
        NS(length=  81, workgroup_size=128, threads_per_transform= 27, factors=(3, 3, 3, 3)),
        NS(length=  84, workgroup_size=128, threads_per_transform= 12, factors=(7, 2, 6)),
        NS(length=  88, workgroup_size=128, threads_per_transform= 11, factors=(11, 8)),
        NS(length=  90, workgroup_size= 64, threads_per_transform=  9, factors=(3, 3, 10)),
        NS(length=  96, workgroup_size=128, threads_per_transform= 16, factors=(6, 16), half_lds=False),
        NS(length= 100, workgroup_size= 64, threads_per_transform= 10, factors=(10, 10)),
        NS(length= 104, workgroup_size= 64, threads_per_transform=  8, factors=(13, 8)),
        NS(length= 108, workgroup_size=256, threads_per_transform= 36, factors=(6, 6, 3)),
        NS(length= 112, workgroup_size=256, threads_per_transform= 16, factors=(16, 7), half_lds=False),
        NS(length= 120, workgroup_size= 64, threads_per_transform= 12, factors=(6, 10, 2)),
        NS(length= 121, workgroup_size=128, threads_per_transform= 11, factors=(11, 11)),
        NS(length= 125, workgroup_size=256, threads_per_transform= 25, factors=(5, 5, 5), half_lds=False),
        NS(length= 128, workgroup_size=256, threads_per_transform= 16, factors=(16, 8)),
        NS(length= 135, workgroup_size=128, threads_per_transform=  9, factors=(5, 3, 3, 3)),
        NS(length= 144, workgroup_size=128, threads_per_transform= 12, factors=(6, 6, 4)),
        NS(length= 150, workgroup_size= 64, threads_per_transform=  5, factors=(10, 5, 3)),
        NS(length= 160, workgroup_size=256, threads_per_transform= 16, factors=(16, 10)),
        NS(length= 162, workgroup_size=256, threads_per_transform= 27, factors=(6, 3, 3, 3)),
        NS(length= 168, workgroup_size=256, threads_per_transform= 56, factors=(8, 7, 3), half_lds=False),
        NS(length= 169, workgroup_size=256, threads_per_transform= 13, factors=(13, 13)),
        NS(length= 176, workgroup_size= 64, threads_per_transform= 16, factors=(11, 16)),
        NS(length= 180, workgroup_size=256, threads_per_transform= 60, factors=(10, 6, 3), half_lds=False),
        NS(length= 192, workgroup_size=128, threads_per_transform= 16, factors=(6, 4, 4, 2)),
        NS(length= 200, workgroup_size= 64, threads_per_transform= 20, factors=(10, 10, 2)),
        NS(length= 208, workgroup_size= 64, threads_per_transform= 16, factors=(13, 16)),
        NS(length= 216, workgroup_size=256, threads_per_transform= 36, factors=(6, 6, 6)),
        NS(length= 224, workgroup_size= 64, threads_per_transform= 16, factors=(7, 2, 2, 2, 2, 2)),
        NS(length= 225, workgroup_size=256, threads_per_transform= 75, factors=(5, 5, 3, 3)),
        NS(length= 240, workgroup_size=128, threads_per_transform= 48, factors=(8, 5, 6)),
        NS(length= 243, workgroup_size=256, threads_per_transform= 81, factors=(3, 3, 3, 3, 3)),
        NS(length= 250, workgroup_size=128, threads_per_transform= 25, factors=(10, 5, 5)),
        NS(length= 256, workgroup_size= 64, threads_per_transform= 64, factors=(4, 4, 4, 4)),
        NS(length= 270, workgroup_size=128, threads_per_transform= 27, factors=(10, 3, 3, 3)),
        NS(length= 272, workgroup_size=128, threads_per_transform= 17, factors=(16, 17)),
        NS(length= 288, workgroup_size=128, threads_per_transform= 24, factors=(6, 6, 4, 2)),
        NS(length= 300, workgroup_size= 64, threads_per_transform= 30, factors=(10, 10, 3)),
        NS(length= 320, workgroup_size= 64, threads_per_transform= 16, factors=(10, 4, 4, 2)),
        NS(length= 324, workgroup_size= 64, threads_per_transform= 54, factors=(3, 6, 6, 3)),
        NS(length= 336, workgroup_size=128, threads_per_transform= 56, factors=(8, 7, 6)),
        NS(length= 343, workgroup_size=256, threads_per_transform= 49, factors=(7, 7, 7)),
        NS(length= 360, workgroup_size=256, threads_per_transform= 60, factors=(10, 6, 6)),
        NS(length= 375, workgroup_size=128, threads_per_transform= 25, factors=(5, 5, 5, 3)),
        NS(length= 384, workgroup_size=128, threads_per_transform= 32, factors=(6, 4, 4, 4)),
        NS(length= 400, workgroup_size=128, threads_per_transform= 40, factors=(4, 10, 10)),
        NS(length= 405, workgroup_size=128, threads_per_transform= 27, factors=(5, 3, 3, 3, 3)),
        NS(length= 432, workgroup_size= 64, threads_per_transform= 27, factors=(3, 16, 3, 3)),
        NS(length= 450, workgroup_size=128, threads_per_transform= 30, factors=(10, 5, 3, 3)),
        NS(length= 480, workgroup_size= 64, threads_per_transform= 16, factors=(10, 8, 6)),
        NS(length= 486, workgroup_size=256, threads_per_transform=162, factors=(6, 3, 3, 3, 3)),
        NS(length= 500, workgroup_size=128, threads_per_transform=100, factors=(10, 5, 10)),
        NS(length= 512, workgroup_size= 64, threads_per_transform= 64, factors=(8, 8, 8)),
        NS(length= 528, workgroup_size= 64, threads_per_transform= 48, factors=(4, 4, 3, 11)),
        NS(length= 540, workgroup_size=256, threads_per_transform= 54, factors=(3, 10, 6, 3)),
        NS(length= 576, workgroup_size=128, threads_per_transform= 96, factors=(16, 6, 6)),
        NS(length= 600, workgroup_size= 64, threads_per_transform= 60, factors=(10, 6, 10)),
        NS(length= 625, workgroup_size=128, threads_per_transform=125, factors=(5, 5, 5, 5)),
        NS(length= 640, workgroup_size=128, threads_per_transform= 64, factors=(8, 10, 8)),
        NS(length= 648, workgroup_size=256, threads_per_transform=216, factors=(8, 3, 3, 3, 3)),
        NS(length= 675, workgroup_size=256, threads_per_transform=225, factors=(5, 5, 3, 3, 3)),
        NS(length= 720, workgroup_size=256, threads_per_transform=120, factors=(10, 3, 8, 3)),
        NS(length= 729, workgroup_size=256, threads_per_transform=243, factors=(3, 3, 3, 3, 3, 3)),
        NS(length= 750, workgroup_size=256, threads_per_transform=250, factors=(10, 5, 3, 5)),
        NS(length= 768, workgroup_size= 64, threads_per_transform= 48, factors=(16, 3, 16)),
        NS(length= 800, workgroup_size=256, threads_per_transform=160, factors=(16, 5, 10)),
        NS(length= 810, workgroup_size=128, threads_per_transform= 81, factors=(3, 10, 3, 3, 3)),
        NS(length= 864, workgroup_size= 64, threads_per_transform= 54, factors=(3, 6, 16, 3)),
        NS(length= 900, workgroup_size=256, threads_per_transform= 90, factors=(10, 10, 3, 3)),
        NS(length= 960, workgroup_size=256, threads_per_transform=160, factors=(16, 10, 6), half_lds=False),
        NS(length= 972, workgroup_size=256, threads_per_transform=162, factors=(3, 6, 3, 6, 3)),
        NS(length=1000, workgroup_size=128, threads_per_transform=100, factors=(10, 10, 10)),
        NS(length=1024, workgroup_size=128, threads_per_transform=128, factors=(8, 8, 4, 4)),
        NS(length=1040, workgroup_size=256, threads_per_transform=208, factors=(13, 16, 5), runtime_compile=True),
        NS(length=1080, workgroup_size=256, threads_per_transform=108, factors=(6, 10, 6, 3), runtime_compile=True),
        NS(length=1125, workgroup_size=256, threads_per_transform=225, factors=(5, 5, 3, 3, 5), runtime_compile=True),
        NS(length=1152, workgroup_size=256, threads_per_transform=144, factors=(4, 3, 8, 3, 4), runtime_compile=True),
        NS(length=1200, workgroup_size=256, threads_per_transform= 75, factors=(5, 5, 16, 3), runtime_compile=True),
        NS(length=1215, workgroup_size=256, threads_per_transform=243, factors=(5, 3, 3, 3, 3, 3), runtime_compile=True),
        NS(length=1250, workgroup_size=256, threads_per_transform=250, factors=(5, 10, 5, 5), runtime_compile=True),
        NS(length=1280, workgroup_size=128, threads_per_transform= 80, factors=(16, 5, 16), runtime_compile=True),
        NS(length=1296, workgroup_size=128, threads_per_transform=108, factors=(6, 6, 6, 6), runtime_compile=True),
        NS(length=1350, workgroup_size=256, threads_per_transform=135, factors=(5, 10, 3, 3, 3), runtime_compile=True),
        NS(length=1440, workgroup_size=128, threads_per_transform= 90, factors=(10, 16, 3, 3), runtime_compile=True),
        NS(length=1458, workgroup_size=256, threads_per_transform=243, factors=(6, 3, 3, 3, 3, 3), runtime_compile=True),
        NS(length=1500, workgroup_size=256, threads_per_transform=150, factors=(5, 10, 10, 3), runtime_compile=True),
        NS(length=1536, workgroup_size=256, threads_per_transform=256, factors=(16, 16, 6), runtime_compile=True),
        NS(length=1600, workgroup_size=256, threads_per_transform=100, factors=(10, 16, 10), runtime_compile=True),
        NS(length=1620, workgroup_size=256, threads_per_transform=162, factors=(10, 3, 3, 6, 3), runtime_compile=True),
        NS(length=1728, workgroup_size=128, threads_per_transform=108, factors=(3, 6, 6, 16), runtime_compile=True),
        NS(length=1800, workgroup_size=256, threads_per_transform=180, factors=(10, 6, 10, 3), runtime_compile=True),
        NS(length=1875, workgroup_size=256, threads_per_transform=125, factors=(5, 5, 5, 5, 3), runtime_compile=True),
        NS(length=1920, workgroup_size=256, threads_per_transform=120, factors=(10, 6, 16, 2), runtime_compile=True),
        NS(length=1944, workgroup_size=256, threads_per_transform=243, factors=(3, 3, 3, 3, 8, 3), runtime_compile=True),
        NS(length=2000, workgroup_size=128, threads_per_transform=125, factors=(5, 5, 5, 16), runtime_compile=True),
        NS(length=2025, workgroup_size=256, threads_per_transform=135, factors=(3, 3, 5, 5, 3, 3), runtime_compile=True),
        NS(length=2048, workgroup_size=256, threads_per_transform=256, factors=(16, 16, 8), runtime_compile=True),
        NS(length=2160, workgroup_size=256, threads_per_transform= 60, factors=(10, 6, 6, 6), runtime_compile=True),
        NS(length=2187, workgroup_size=256, threads_per_transform=243, factors=(3, 3, 3, 3, 3, 3, 3), runtime_compile=True),
        NS(length=2250, workgroup_size=256, threads_per_transform= 90, factors=(10, 3, 5, 3, 5), runtime_compile=True),
        NS(length=2304, workgroup_size=256, threads_per_transform=192, factors=(6, 6, 4, 4, 4), runtime_compile=True),
        NS(length=2400, workgroup_size=256, threads_per_transform=240, factors=(4, 10, 10, 6), runtime_compile=True),
        NS(length=2430, workgroup_size=256, threads_per_transform= 81, factors=(10, 3, 3, 3, 3, 3), runtime_compile=True),
        NS(length=2500, workgroup_size=256, threads_per_transform=250, factors=(10, 5, 10, 5), runtime_compile=True),
        NS(length=2560, workgroup_size=128, threads_per_transform=128, factors=(4, 4, 4, 10, 4), runtime_compile=True),
        NS(length=2592, workgroup_size=256, threads_per_transform=216, factors=(6, 6, 6, 6, 2), runtime_compile=True),
        NS(length=2700, workgroup_size=128, threads_per_transform= 90, factors=(3, 10, 10, 3, 3), runtime_compile=True),
        NS(length=2880, workgroup_size=256, threads_per_transform= 96, factors=(10, 6, 6, 2, 2, 2), runtime_compile=True),
        NS(length=2916, workgroup_size=256, threads_per_transform=243, factors=(6, 6, 3, 3, 3, 3), runtime_compile=True),
        NS(length=3000, workgroup_size=128, threads_per_transform=100, factors=(10, 3, 10, 10), runtime_compile=True),
        NS(length=3072, workgroup_size=256, threads_per_transform=256, factors=(6, 4, 4, 4, 4, 2), runtime_compile=True),
        NS(length=3125, workgroup_size=128, threads_per_transform=125, factors=(5, 5, 5, 5, 5), runtime_compile=True),
        NS(length=3200, workgroup_size=256, threads_per_transform=160, factors=(10, 10, 4, 4, 2), runtime_compile=True),
        NS(length=3240, workgroup_size=128, threads_per_transform=108, factors=(3, 3, 10, 6, 6), runtime_compile=True),
        NS(length=3375, workgroup_size=256, threads_per_transform=225, factors=(5, 5, 5, 3, 3, 3), runtime_compile=True),
        NS(length=3456, workgroup_size=256, threads_per_transform=144, factors=(6, 6, 6, 4, 4), runtime_compile=True),
        NS(length=3600, workgroup_size=256, threads_per_transform=120, factors=(10, 10, 6, 6), runtime_compile=True),
        NS(length=3645, workgroup_size=256, threads_per_transform=243, factors=(5, 3, 3, 3, 3, 3, 3), runtime_compile=True),
        NS(length=3750, workgroup_size=256, threads_per_transform=125, factors=(3, 5, 5, 10, 5), runtime_compile=True),
        NS(length=3840, workgroup_size=256, threads_per_transform=128, factors=(10, 6, 2, 2, 2, 2, 2, 2), runtime_compile=True),
        NS(length=3888, workgroup_size=512, threads_per_transform=324, factors=(16, 3, 3, 3, 3, 3), runtime_compile=True),
        NS(length=4000, workgroup_size=256, threads_per_transform=200, factors=(10, 10, 10, 4), runtime_compile=True),
        NS(length=4050, workgroup_size=256, threads_per_transform=135, factors=(10, 5, 3, 3, 3, 3), runtime_compile=True),
        NS(length=4096, workgroup_size=256, threads_per_transform=256, factors=(16, 16, 16), runtime_compile=True),
    ]

    kernels = [NS(**kernel.__dict__,
                  scheme='CS_KERNEL_STOCKHAM',
                  precision=['sp', 'dp']) for kernel in kernels1d]

    return kernels

def list_2d_kernels():
    """Return list of fused 2D kernels to generate."""

    fused_kernels = [
        NS(length=[4,4], factors=[[2,2],[2,2]], threads_per_transform=[2,2], workgroup_size=8),
        NS(length=[4,8], factors=[[2,2],[4,2]], threads_per_transform=[2,2], workgroup_size=16),
        NS(length=[4,9], factors=[[2,2],[3,3]], threads_per_transform=[2,3], workgroup_size=18),
        NS(length=[4,16], factors=[[2,2],[4,4]], threads_per_transform=[2,4], workgroup_size=32),
        NS(length=[4,25], factors=[[2,2],[5,5]], threads_per_transform=[2,5], workgroup_size=50),
        NS(length=[4,27], factors=[[2,2],[3,3,3]], threads_per_transform=[2,9], workgroup_size=54),
        NS(length=[4,32], factors=[[2,2],[8,4]], threads_per_transform=[2,4], workgroup_size=64),
        NS(length=[4,64], factors=[[2,2],[4,4,4]], threads_per_transform=[2,16], workgroup_size=128),
        NS(length=[4,81], factors=[[2,2],[3,3,3,3]], threads_per_transform=[2,27], workgroup_size=162),
        NS(length=[4,125], factors=[[2,2],[5,5,5]], threads_per_transform=[2,25], workgroup_size=250),
        NS(length=[4,128], factors=[[2,2],[8,4,4]], threads_per_transform=[2,16], workgroup_size=256),
        NS(length=[4,243], factors=[[2,2],[3,3,3,3,3]], threads_per_transform=[2,81], workgroup_size=486),
        NS(length=[4,256], factors=[[2,2],[4,4,4,4]], threads_per_transform=[2,64], workgroup_size=512),
        NS(length=[8,4], factors=[[4,2],[2,2]], threads_per_transform=[2,2], workgroup_size=16),
        NS(length=[8,8], factors=[[4,2],[4,2]], threads_per_transform=[2,2], workgroup_size=16),
        NS(length=[8,9], factors=[[4,2],[3,3]], threads_per_transform=[2,3], workgroup_size=24),
        NS(length=[8,16], factors=[[4,2],[4,4]], threads_per_transform=[2,4], workgroup_size=32),
        NS(length=[8,25], factors=[[4,2],[5,5]], threads_per_transform=[2,5], workgroup_size=50),
        NS(length=[8,27], factors=[[4,2],[3,3,3]], threads_per_transform=[2,9], workgroup_size=72),
        NS(length=[8,32], factors=[[4,2],[8,4]], threads_per_transform=[2,4], workgroup_size=64),
        NS(length=[8,64], factors=[[4,2],[4,4,4]], threads_per_transform=[2,16], workgroup_size=128),
        NS(length=[8,81], factors=[[4,2],[3,3,3,3]], threads_per_transform=[2,27], workgroup_size=216),
        NS(length=[8,125], factors=[[4,2],[5,5,5]], threads_per_transform=[2,25], workgroup_size=250),
        NS(length=[8,128], factors=[[4,2],[8,4,4]], threads_per_transform=[2,16], workgroup_size=256),
        NS(length=[8,243], factors=[[4,2],[3,3,3,3,3]], threads_per_transform=[2,81], workgroup_size=648),
        NS(length=[8,256], factors=[[4,2],[4,4,4,4]], threads_per_transform=[2,64], workgroup_size=512),
        NS(length=[9,4], factors=[[3,3],[2,2]], threads_per_transform=[3,2], workgroup_size=18),
        NS(length=[9,8], factors=[[3,3],[4,2]], threads_per_transform=[3,2], workgroup_size=24),
        NS(length=[9,9], factors=[[3,3],[3,3]], threads_per_transform=[3,3], workgroup_size=27),
        NS(length=[9,16], factors=[[3,3],[4,4]], threads_per_transform=[3,4], workgroup_size=48),
        NS(length=[9,25], factors=[[3,3],[5,5]], threads_per_transform=[3,5], workgroup_size=75),
        NS(length=[9,27], factors=[[3,3],[3,3,3]], threads_per_transform=[3,9], workgroup_size=81),
        NS(length=[9,32], factors=[[3,3],[8,4]], threads_per_transform=[3,4], workgroup_size=96),
        NS(length=[9,64], factors=[[3,3],[4,4,4]], threads_per_transform=[3,16], workgroup_size=192),
        NS(length=[9,81], factors=[[3,3],[3,3,3,3]], threads_per_transform=[3,27], workgroup_size=243),
        NS(length=[9,125], factors=[[3,3],[5,5,5]], threads_per_transform=[3,25], workgroup_size=375),
        NS(length=[9,128], factors=[[3,3],[8,4,4]], threads_per_transform=[3,16], workgroup_size=384),
        NS(length=[9,243], factors=[[3,3],[3,3,3,3,3]], threads_per_transform=[3,81], workgroup_size=729),
        NS(length=[9,256], factors=[[3,3],[4,4,4,4]], threads_per_transform=[3,64], workgroup_size=768),
        NS(length=[16,4], factors=[[4,4],[2,2]], threads_per_transform=[4,2], workgroup_size=32),
        NS(length=[16,8], factors=[[4,4],[4,2]], threads_per_transform=[4,2], workgroup_size=32),
        NS(length=[16,9], factors=[[4,4],[3,3]], threads_per_transform=[4,3], workgroup_size=48),
        NS(length=[16,16], factors=[[4,4],[4,4]], threads_per_transform=[4,4], workgroup_size=64),
        NS(length=[16,25], factors=[[4,4],[5,5]], threads_per_transform=[4,5], workgroup_size=100),
        NS(length=[16,27], factors=[[4,4],[3,3,3]], threads_per_transform=[4,9], workgroup_size=144),
        NS(length=[16,32], factors=[[4,4],[8,4]], threads_per_transform=[4,4], workgroup_size=128),
        NS(length=[16,64], factors=[[4,4],[4,4,4]], threads_per_transform=[4,16], workgroup_size=256),
        NS(length=[16,81], factors=[[4,4],[3,3,3,3]], threads_per_transform=[4,27], workgroup_size=432),
        NS(length=[16,125], factors=[[4,4],[5,5,5]], threads_per_transform=[4,25], workgroup_size=500),
        NS(length=[16,128], factors=[[4,4],[8,4,4]], threads_per_transform=[4,16], workgroup_size=512),
        NS(length=[25,4], factors=[[5,5],[2,2]], threads_per_transform=[5,2], workgroup_size=50),
        NS(length=[25,8], factors=[[5,5],[4,2]], threads_per_transform=[5,2], workgroup_size=50),
        NS(length=[25,9], factors=[[5,5],[3,3]], threads_per_transform=[5,3], workgroup_size=75),
        NS(length=[25,16], factors=[[5,5],[4,4]], threads_per_transform=[5,4], workgroup_size=100),
        NS(length=[25,25], factors=[[5,5],[5,5]], threads_per_transform=[5,5], workgroup_size=125),
        NS(length=[25,27], factors=[[5,5],[3,3,3]], threads_per_transform=[5,9], workgroup_size=225),
        NS(length=[25,32], factors=[[5,5],[8,4]], threads_per_transform=[5,4], workgroup_size=160),
        NS(length=[25,64], factors=[[5,5],[4,4,4]], threads_per_transform=[5,16], workgroup_size=400),
        NS(length=[25,81], factors=[[5,5],[3,3,3,3]], threads_per_transform=[5,27], workgroup_size=675),
        NS(length=[25,125], factors=[[5,5],[5,5,5]], threads_per_transform=[5,25], workgroup_size=625),
        NS(length=[25,128], factors=[[5,5],[8,4,4]], threads_per_transform=[5,16], workgroup_size=640),
        NS(length=[27,4], factors=[[3,3,3],[2,2]], threads_per_transform=[9,2], workgroup_size=54),
        NS(length=[27,8], factors=[[3,3,3],[4,2]], threads_per_transform=[9,2], workgroup_size=72),
        NS(length=[27,9], factors=[[3,3,3],[3,3]], threads_per_transform=[9,3], workgroup_size=81),
        NS(length=[27,16], factors=[[3,3,3],[4,4]], threads_per_transform=[9,4], workgroup_size=144),
        NS(length=[27,25], factors=[[3,3,3],[5,5]], threads_per_transform=[9,5], workgroup_size=225),
        NS(length=[27,27], factors=[[3,3,3],[3,3,3]], threads_per_transform=[9,9], workgroup_size=243),
        NS(length=[27,32], factors=[[3,3,3],[8,4]], threads_per_transform=[9,4], workgroup_size=288),
        NS(length=[27,64], factors=[[3,3,3],[4,4,4]], threads_per_transform=[9,16], workgroup_size=576),
        NS(length=[27,81], factors=[[3,3,3],[3,3,3,3]], threads_per_transform=[9,27], workgroup_size=729),
        NS(length=[32,4], factors=[[8,4],[2,2]], threads_per_transform=[4,2], workgroup_size=64),
        NS(length=[32,8], factors=[[8,4],[4,2]], threads_per_transform=[4,2], workgroup_size=64),
        NS(length=[32,9], factors=[[8,4],[3,3]], threads_per_transform=[4,3], workgroup_size=96),
        NS(length=[32,16], factors=[[8,4],[4,4]], threads_per_transform=[4,4], workgroup_size=128),
        NS(length=[32,25], factors=[[8,4],[5,5]], threads_per_transform=[4,5], workgroup_size=160),
        NS(length=[32,27], factors=[[8,4],[3,3,3]], threads_per_transform=[4,9], workgroup_size=288),
        NS(length=[32,32], factors=[[8,4],[8,4]], threads_per_transform=[4,4], workgroup_size=128),
        NS(length=[32,64], factors=[[8,4],[4,4,4]], threads_per_transform=[4,16], workgroup_size=512),
        NS(length=[32,81], factors=[[8,4],[3,3,3,3]], threads_per_transform=[4,27], workgroup_size=864),
        NS(length=[32,125], factors=[[8,4],[5,5,5]], threads_per_transform=[4,25], workgroup_size=800),
        NS(length=[32,128], factors=[[8,4],[8,4,4]], threads_per_transform=[4,16], workgroup_size=512),
        NS(length=[64,4], factors=[[4,4,4],[2,2]], threads_per_transform=[16,2], workgroup_size=128),
        NS(length=[64,8], factors=[[4,4,4],[4,2]], threads_per_transform=[16,2], workgroup_size=128),
        NS(length=[64,9], factors=[[4,4,4],[3,3]], threads_per_transform=[16,3], workgroup_size=192),
        NS(length=[64,16], factors=[[4,4,4],[4,4]], threads_per_transform=[16,4], workgroup_size=256),
        NS(length=[64,25], factors=[[4,4,4],[5,5]], threads_per_transform=[16,5], workgroup_size=400),
        NS(length=[64,27], factors=[[4,4,4],[3,3,3]], threads_per_transform=[16,9], workgroup_size=576),
        NS(length=[64,32], factors=[[4,4,4],[8,4]], threads_per_transform=[16,4], workgroup_size=512),
        NS(length=[81,4], factors=[[3,3,3,3],[2,2]], threads_per_transform=[27,2], workgroup_size=162),
        NS(length=[81,8], factors=[[3,3,3,3],[4,2]], threads_per_transform=[27,2], workgroup_size=216),
        NS(length=[81,9], factors=[[3,3,3,3],[3,3]], threads_per_transform=[27,3], workgroup_size=243),
        NS(length=[81,16], factors=[[3,3,3,3],[4,4]], threads_per_transform=[27,4], workgroup_size=432),
        NS(length=[81,25], factors=[[3,3,3,3],[5,5]], threads_per_transform=[27,5], workgroup_size=675),
        NS(length=[81,27], factors=[[3,3,3,3],[3,3,3]], threads_per_transform=[27,9], workgroup_size=729),
        NS(length=[81,32], factors=[[3,3,3,3],[8,4]], threads_per_transform=[27,4], workgroup_size=864),
        NS(length=[125,4], factors=[[5,5,5],[2,2]], threads_per_transform=[25,2], workgroup_size=250),
        NS(length=[125,8], factors=[[5,5,5],[4,2]], threads_per_transform=[25,2], workgroup_size=250),
        NS(length=[125,9], factors=[[5,5,5],[3,3]], threads_per_transform=[25,3], workgroup_size=375),
        NS(length=[125,16], factors=[[5,5,5],[4,4]], threads_per_transform=[25,4], workgroup_size=500),
        NS(length=[125,25], factors=[[5,5,5],[5,5]], threads_per_transform=[25,5], workgroup_size=625),
        NS(length=[125,32], factors=[[5,5,5],[8,4]], threads_per_transform=[25,4], workgroup_size=800),
        NS(length=[128,4], factors=[[8,4,4],[2,2]], threads_per_transform=[16,2], workgroup_size=256),
        NS(length=[128,8], factors=[[8,4,4],[4,2]], threads_per_transform=[16,2], workgroup_size=256),
        NS(length=[128,9], factors=[[8,4,4],[3,3]], threads_per_transform=[16,3], workgroup_size=384),
        NS(length=[128,16], factors=[[8,4,4],[4,4]], threads_per_transform=[16,4], workgroup_size=512),
        NS(length=[128,25], factors=[[8,4,4],[5,5]], threads_per_transform=[16,5], workgroup_size=640),
        NS(length=[128,32], factors=[[8,4,4],[8,4]], threads_per_transform=[16,4], workgroup_size=512),
        NS(length=[243,4], factors=[[3,3,3,3,3],[2,2]], threads_per_transform=[81,2], workgroup_size=486),
        NS(length=[243,8], factors=[[3,3,3,3,3],[4,2]], threads_per_transform=[81,2], workgroup_size=648),
        NS(length=[243,9], factors=[[3,3,3,3,3],[3,3]], threads_per_transform=[81,3], workgroup_size=729),
        NS(length=[256,4], factors=[[4,4,4,4],[2,2]], threads_per_transform=[64,2], workgroup_size=512),
        NS(length=[256,8], factors=[[4,4,4,4],[4,2]], threads_per_transform=[64,2], workgroup_size=512),
        NS(length=[256,9], factors=[[4,4,4,4],[3,3]], threads_per_transform=[64,3], workgroup_size=768),
    ]

    expanded = []
    expanded.extend(NS(**kernel.__dict__,
                       scheme='CS_KERNEL_2D_SINGLE') for kernel in fused_kernels)

    return expanded


def list_large_kernels():
    """Return list of large kernels to generate."""

    sbcc_kernels = [
        NS(length=50,  factors=[10, 5],      use_3steps_large_twd={
           'sp': 'true',  'dp': 'true'}, workgroup_size=256),
        NS(length=52,  factors=[13, 4],      use_3steps_large_twd={
           'sp': 'true',  'dp': 'true'}),
        NS(length=60,  factors=[6, 10],      use_3steps_large_twd={
           'sp': 'false',  'dp': 'false'}),
        NS(length=64,  factors=[8, 8],       use_3steps_large_twd={
           'sp': 'true',  'dp': 'false'}),
        NS(length=72,  factors=[8, 3, 3],    use_3steps_large_twd={
           'sp': 'true',  'dp': 'false'}),
        NS(length=80,  factors=[10, 8],      use_3steps_large_twd={
           'sp': 'false',  'dp': 'false'}),
        NS(length=81,  factors=[3, 3, 3, 3], use_3steps_large_twd={
           'sp': 'true',  'dp': 'true'}),
        NS(length=84,  factors=[7, 2, 6],    use_3steps_large_twd={
           'sp': 'true',  'dp': 'true'}),
        NS(length=96,  factors=[8, 3, 4],    use_3steps_large_twd={
           'sp': 'false',  'dp': 'false'}, workgroup_size=256),
        NS(length=100, factors=[5, 5, 4],    use_3steps_large_twd={
           'sp': 'true',  'dp': 'false'}, workgroup_size=100),
        NS(length=104, factors=[13, 8],      use_3steps_large_twd={
           'sp': 'true',  'dp': 'false'}),
        NS(length=108, factors=[6, 6, 3],    use_3steps_large_twd={
           'sp': 'true',  'dp': 'false'}),
        NS(length=112, factors=[4, 7, 4],    use_3steps_large_twd={
           'sp': 'false',  'dp': 'false'}),
        NS(length=128, factors=[8, 4, 4],    use_3steps_large_twd={
           'sp': 'true',  'dp': 'true'}, workgroup_size=256),
        NS(length=160, factors=[4, 10, 4],   use_3steps_large_twd={
           'sp': 'false', 'dp': 'false'}, flavour='wide'),
        NS(length=168, factors=[7, 6, 4],    use_3steps_large_twd={
           'sp': 'true', 'dp': 'false'}, workgroup_size=128),
        # NS(length=192, factors=[6, 4, 4, 2], use_3steps_large_twd={
        #    'sp': 'false', 'dp': 'false'}),
        NS(length=200, factors=[8, 5, 5],    use_3steps_large_twd={
           'sp': 'false', 'dp': 'false'}),
        NS(length=208, factors=[13, 16],     use_3steps_large_twd={
           'sp': 'false', 'dp': 'false'}),
        NS(length=216, factors=(6, 6, 6), use_3steps_large_twd={
           'sp': 'false', 'dp': 'false'}, threads_per_transform= 36),
        NS(length=224, factors=[8, 7, 4],    use_3steps_large_twd={
           'sp': 'true', 'dp': 'false'}),
        NS(length=240, factors=[8, 5, 6],    use_3steps_large_twd={
           'sp': 'false', 'dp': 'false'}),
        NS(length=256, factors=[8, 4, 8], use_3steps_large_twd={
           'sp': 'true',  'dp': 'false'}, flavour='wide'),
        NS(length=336, factors=[6, 7, 8],    use_3steps_large_twd={
           'sp': 'false', 'dp': 'false'}),
        NS(length=343, factors=[7, 7, 7],    use_3steps_large_twd={
           'sp': 'true', 'dp': 'true'}),
        NS(length=512, factors=[8, 8, 8],    use_3steps_large_twd={
           'sp': 'true', 'dp': 'false'}),
    ]

    # for SBCC kernel, increase desired workgroup_size so that columns per
    # thread block is also increased. currently targeting for 16 columns
    block_width = 16
    for k in sbcc_kernels:
        k.scheme = 'CS_KERNEL_STOCKHAM_BLOCK_CC'
        if not hasattr(k, 'workgroup_size'):
            k.workgroup_size = block_width * \
                functools.reduce(mul, k.factors, 1) // min(k.factors)
        if not hasattr(k, 'length'):
            k.length = functools.reduce(lambda a, b: a * b, k.factors)

    # SBRC
    sbrc_kernels = [
        NS(length=49,  factors=[7, 7], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=196, threads_per_transform=7), # block_width=28
        NS(length=50,  factors=[10, 5], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=50, threads_per_transform=5), # block_width=10
        # SBRC64: wgs=256 poor in MI50
        NS(length=64,  factors=[4, 4, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=128, threads_per_transform=16), # block_width=8
        # NS(length=81,  factors=[3, 3, 3, 3], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=486, threads_per_transform=27), # block_width=18
        NS(length=81,  factors=[3, 3, 3, 3], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=243, threads_per_transform=27), # block_width=9
        NS(length=100, factors=[5, 5, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=100, threads_per_transform=25), # block_width=4
        NS(length=112, factors=[4, 7, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=448, threads_per_transform=28), # block_width=16
        NS(length=128, factors=[8, 4, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=128, threads_per_transform=16), # block_width=8
        # NS(length=128, factors=[8, 4, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=32), # block_width=8
        NS(length=192, factors=[6, 4, 4, 2], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=32), # block_width=8
        NS(length=200, factors=[8, 5, 5], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=400, threads_per_transform=40), # block_width=10
        NS(length=256, factors=[4, 4, 4, 4], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=256, threads_per_transform=32), # block_width=8
        NS(length=512, factors=[8, 8, 8], scheme='CS_KERNEL_STOCKHAM_BLOCK_RC', workgroup_size=512, threads_per_transform=128),
    ]

    # NB:
    # Technically, we could have SBCR kernels the same amount as SBCC.
    #
    # sbcr_kernels = copy.deepcopy(sbcc_kernels)
    # for k in sbcr_kernels:
    #     k.scheme = 'CS_KERNEL_STOCKHAM_BLOCK_CR'
    #

    sbcr_kernels = [
        NS(length=56,  factors=[7, 8],  use_3steps_large_twd={'sp': 'false', 'dp': 'false'}),
        NS(length=100, factors=[10, 10],    use_3steps_large_twd={
           'sp': 'true',  'dp': 'false'}, workgroup_size=100),
        NS(length=200, factors=[8, 5, 5],    use_3steps_large_twd={
           'sp': 'false', 'dp': 'false'}),
        NS(length=336, factors=[6, 7, 8],    use_3steps_large_twd={
           'sp': 'false', 'dp': 'false'})
    ]

    block_width = 16
    for k in sbcr_kernels:
        k.scheme = 'CS_KERNEL_STOCKHAM_BLOCK_CR'
        if not hasattr(k, 'workgroup_size'):
            k.workgroup_size = block_width * \
                functools.reduce(mul, k.factors, 1) // min(k.factors)
        if not hasattr(k, 'length'):
            k.length = functools.reduce(lambda a, b: a * b, k.factors)

    return sbcc_kernels + sbcr_kernels + sbrc_kernels


def default_runtime_compile(kernels):
    '''Returns a copy of input kernel list with a default value for runtime_compile.'''

    return [k if hasattr(k, 'runtime_compile') else NS(**k.__dict__, runtime_compile=False) for k in kernels]

def generate_kernel(kernel, precisions, stockham_aot):
    """Generate a single kernel file for 'kernel'.

    The kernel file contains all kernel variations corresponding to
    the kernel meta data in 'kernel'.

    A list of CPU functions is returned.
    """

    args = [stockham_aot]
    pre_enum = { 'sp': 0, 'dp': 1 }
    # 2D single kernels always specify threads per transform
    if isinstance(kernel.length, list):
        args.append(','.join([str(f) for f in kernel.factors[0]]))
        args.append(','.join([str(f) for f in kernel.factors[1]]))
        args.append(','.join([str(pre_enum[pre]) for pre in precisions]))
        args.append(','.join([str(f) for f in kernel.threads_per_transform]))
    else:
        args.append(','.join([str(f) for f in kernel.factors]))
        args.append(','.join([str(pre_enum[pre]) for pre in precisions]))
        # 1D kernels might not, and need to default to 'uwide'
        threads_per_transform = getattr(kernel,'threads_per_transform', {
            'uwide': kernel.length // min(kernel.factors),
            'wide': kernel.length // max(kernel.factors),
            'tall': 0,
            'consolidated': 0
            }[getattr(kernel,'flavour', 'uwide')])
        args.append(str(threads_per_transform))

    # default half_lds to True only for CS_KERNEL_STOCKHAM
    half_lds = getattr(kernel, 'half_lds', kernel.scheme == 'CS_KERNEL_STOCKHAM')

    filename = kernel_file_name(kernel)

    args.append(str(kernel.workgroup_size))
    args.append('1' if half_lds else '0')
    args.append(kernel.scheme)
    args.append(filename)

    proc = subprocess.run(args=args, stdout=subprocess.PIPE, check=True)
    clang_format_file(filename)

    import json
    launchers = json.loads(proc.stdout.decode('ascii'))

    cpu_functions = []
    data = Variable('data_p', 'const void *')
    back = Variable('back_p', 'void *')
    for launcher_dict in launchers:
        launcher = NS(**launcher_dict)

        factors = launcher.factors
        length = launcher.lengths[0] if len(launcher.lengths) == 1 else (launcher.lengths[0], launcher.lengths[1])
        transforms_per_block = launcher.transforms_per_block
        workgroup_size = launcher.workgroup_size
        threads_per_transform = workgroup_size // transforms_per_block
        half_lds = launcher.half_lds
        scheme = launcher.scheme
        sbrc_type = launcher.sbrc_type
        sbrc_transpose_type = launcher.sbrc_transpose_type
        precision = 'dp' if launcher.double_precision else 'sp'
        runtime_compile = kernel.runtime_compile
        use_3steps_large_twd = getattr(kernel, 'use_3steps_large_twd', None)

        params = LaunchParams(transforms_per_block, workgroup_size, threads_per_transform, half_lds)

        # make 2D list of threads_per_transform to populate FFTKernel
        tpt_list = kernel.threads_per_transform if scheme == 'CS_KERNEL_2D_SINGLE' else [threads_per_transform, 0]

        f = Function(name=launcher.name,
                     arguments=ArgumentList(data, back),
                     meta=NS(
                         factors=factors,
                         length=length,
                         params=params,
                         precision=precision,
                         runtime_compile=runtime_compile,
                         scheme=scheme,
                         workgroup_size=workgroup_size,
                         transforms_per_block=transforms_per_block,
                         threads_per_transform=tpt_list,
                         transpose=sbrc_transpose_type,
                         use_3steps_large_twd=use_3steps_large_twd,
                         ))

        cpu_functions.append(f)

    return cpu_functions


def generate_kernels(kernels, precisions, stockham_aot):
    """Generate and write kernels from the kernel list.

    Entries in the kernel list are simple namespaces.  These are
    passed as keyword arguments to the Stockham generator.

    A list of CPU functions is returned.
    """
    import threading
    import queue

    # push all the work to a queue
    q_in = queue.Queue()
    for k in kernels:
        q_in.put(k)

    # queue for outputs
    q_out = queue.Queue()

    def threadfunc():
        nonlocal q_in
        nonlocal q_out
        nonlocal precisions
        nonlocal stockham_aot
        try:
            while not q_in.empty():
                k = q_in.get()
                q_out.put(generate_kernel(k, precisions, stockham_aot))
        except queue.Empty:
            pass

    # by default, start up worker threads.  disable this if you want
    # to use pdb to debug
    use_threads = True

    if use_threads:
        threads = []
        for i in range(os.cpu_count()):
            threads.append(threading.Thread(target=threadfunc))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        threadfunc()

    # iterate over the queue
    def queue_iter(q_out):
        try:
            while not q_out.empty():
                yield q_out.get()
        except queue.Empty:
            pass

    return flatten(queue_iter(q_out))


def cli():
    """Command line interface..."""
    parser = argparse.ArgumentParser(prog='kernel-generator')
    subparsers = parser.add_subparsers(dest='command')
    parser.add_argument('--pattern', type=str, help='Kernel pattern to generate.', default='all')
    parser.add_argument('--precision', type=str, help='Precision to generate.', default='all')
    parser.add_argument('--manual-small', type=str, help='Small kernel sizes to generate.')
    parser.add_argument('--manual-large', type=str, help='Large kernel sizes to generate.')
    parser.add_argument('--runtime-compile', type=str, help='Allow runtime-compiled kernels.')

    list_parser = subparsers.add_parser('list', help='List kernel files that will be generated.')

    generate_parser = subparsers.add_parser('generate', help='Generate kernels.')
    generate_parser.add_argument('stockham_aot', type=str, help='Stockham AOT executable.')

    args = parser.parse_args()

    patterns = args.pattern.split(',')
    precisions = args.precision.split(',')
    if 'all' in precisions:
        precisions = ['dp', 'sp']
    precisions = [{'single': 'sp', 'double': 'dp'}.get(p, p) for p in precisions]

    #
    # kernel list
    #

    kernels = []
    # move 2d out from all, no need to iterate the 2d-kernels for non-2d patterns
    kernels_2d = list_2d_kernels()
    all_kernels = list_small_kernels() + list_large_kernels()

    manual_small, manual_large = [], []
    if args.manual_small:
        manual_small = list(map(int, args.manual_small.split(',')))
    if args.manual_large:
        manual_large = list(map(int, args.manual_large.split(',')))

    if 'all' in patterns and not manual_small and not manual_large:
        kernels += all_kernels + kernels_2d
    if 'pow2' in patterns:
        lengths = [2**x for x in range(13)]
        kernels += [k for k in all_kernels if k.length in lengths]
    if 'pow3' in patterns:
        lengths = [3**x for x in range(8)]
        kernels += [k for k in all_kernels if k.length in lengths]
    if 'pow5' in patterns:
        lengths = [5**x for x in range(6)]
        kernels += [k for k in all_kernels if k.length in lengths]
    if 'pow7' in patterns:
        lengths = [7**x for x in range(5)]
        kernels += [k for k in all_kernels if k.length in lengths]
    if 'small' in patterns:
        schemes = ['CS_KERNEL_STOCKHAM']
        kernels += [k for k in all_kernels if k.scheme in schemes]
    if 'large' in patterns:
        schemes = ['CS_KERNEL_STOCKHAM_BLOCK_CC', 'CS_KERNEL_STOCKHAM_BLOCK_RC', 'CS_KERNEL_STOCKHAM_BLOCK_CR']
        kernels += [k for k in all_kernels if k.scheme in schemes]
    if '2D' in patterns:
        kernels += kernels_2d
    if manual_small:
        schemes = ['CS_KERNEL_STOCKHAM']
        kernels += [k for k in all_kernels if k.length in manual_small and k.scheme in schemes]
    if manual_large:
        schemes = ['CS_KERNEL_STOCKHAM_BLOCK_CC', 'CS_KERNEL_STOCKHAM_BLOCK_RC', 'CS_KERNEL_STOCKHAM_BLOCK_CR']
        kernels += [k for k in all_kernels if k.length in manual_large and k.scheme in schemes]

    kernels = unique(kernels)

    #
    # set runtime compile
    #

    kernels = default_runtime_compile(kernels)
    if args.runtime_compile != 'ON':
        for k in kernels:
            k.runtime_compile = False

    #
    # sub commands
    #

    if args.command == 'list':
        scprint(set(['function_pool.cpp'] + list_generated_kernels(kernels)))

    if args.command == 'generate':
        cpu_functions = generate_kernels(kernels, precisions, args.stockham_aot)
        write('function_pool.cpp', generate_cpu_function_pool(cpu_functions), format=True)


if __name__ == '__main__':
    cli()

