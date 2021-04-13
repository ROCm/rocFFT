#!/usr/bin/env python3
"""rocFFT kernel generator.

Currently this acts as a shim between CMake and the C++ kernel generator.

It accept two sub-commands:
1. list - lists files that will be generated
2. generate - pass arguments down to the old generator

Note that 'small' kernels don't decompose their lengths.

"""

import argparse
import collections
import functools
import itertools
import subprocess
import sys

from pathlib import Path
from types import SimpleNamespace as NS

from generator import (ArgumentList, BaseNode, Call, CommentBlock, ExternC, Function, Include,
                       LineBreak, Map, Pragma, StatementList, Variable, name_args, format_and_write)


import stockham


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

def product(*args):
    """Cartesian product of input iteratables, as a list."""
    return list(itertools.product(*args))


def merge(*ds):
    """Merge dictionaries; last one wins."""
    r = collections.OrderedDict()
    for d in ds:
        r.update(d)
    return r


def pmerge(d, fs):
    """Merge d with dicts of {(length, precision, scheme, pool): f}."""
    r = collections.OrderedDict()
    r.update(d)
    for f in fs:
        r[f.meta.length, f.meta.precision, f.meta.scheme, f.meta.pool] = f
    return r


#
# Supported kernel sizes
#

def supported_small_sizes(precision, pow2=True, pow3=True, pow5=True, commonRadix=True):
    """Return list of 1D small kernels."""

    upper_bound = {
        'sp': 4096,
        'dp': 4096,         # of course this isn't 2048... not sure why (double len 1594323 will fail)
    }

    powers = {
        5: [5**k for k in range(6 if pow5 else 1)],
        3: [3**k for k in range(8 if pow3 else 1)],
        2: [2**k for k in range(13 if pow2 else 1)],
    }

    lengths = [p2 * p3 * p5 for p2, p3, p5 in product(powers[2], powers[3], powers[5])]

    # common radix 7, 11, and 13
    if commonRadix:
        lengths += [7, 14, 21, 28, 42, 49, 56, 84, 112, 168, 224, 336, 343]
        lengths += [11, 22, 44, 88, 121, 176]
        lengths += [13, 26, 52, 104, 169, 208]

    def filter_bound(length):
        return length <= upper_bound[precision]

    filtered = sorted([x for x in set(lengths) if filter_bound(x)])

    return product(filtered, ['CS_KERNEL_STOCKHAM'])


def supported_large_sizes(precision):
    """Return list of 1D large block kernels."""

    lengths = [50, 64, 81, 100, 128, 200, 256]
    return product(lengths, ['CS_KERNEL_STOCKHAM_BLOCK_CC',
                             'CS_KERNEL_STOCKHAM_BLOCK_RC'])


def supported_2d_sizes(precision):
    """Return list of 2D kernels."""

    # for now, mimic order of old generator so diffing is easier
    powers = {
        5: [5**k for k in range(3, 1, -1)],
        3: [3**k for k in range(5, 1, -1)],
        2: [2**k for k in range(8, 1, -1)],
    }

    lengths = []
    for b1, b2 in [(2, 2), (3, 3), (5, 5), (2, 3), (3, 2), (3, 5), (5, 3), (2, 5), (5, 2)]:
        lengths.extend(product(powers[b1], powers[b2]))

    max_lds_size_bytes = 64 * 1024
    bytes_per_element = {'sp': 8, 'dp': 16}[precision]

    def filter_lds(length):
        return length[0] * length[1] * bytes_per_element * 1.5 <= max_lds_size_bytes

    # XXX: query properly
    def filter_threads(length):
        avoid = [(16, 243), (16, 256), (27, 125), (27, 128), (64, 64), (64, 81)]
        rlength = (length[1], length[0])
        return length not in avoid and rlength not in avoid

    filtered = [x for x in lengths if filter_lds(x) and filter_threads(x)]

    return product(filtered, ['CS_KERNEL_2D_SINGLE'])

#
# Prototype generators
#

@name_args(['function'])
class FFTKernel(BaseNode):
    def __str__(self):
        f = 'FFTKernel('
        f += str(self.function.address())
        use_3steps_large_twd = getattr(self.function.meta, 'use_3steps_large_twd', None)
        if use_3steps_large_twd is not None:
            f += ', ' + str(use_3steps_large_twd)
        factors = getattr(self.function.meta, 'factors', None)
        if factors is not None:
            f += ', {' + cjoin(factors) + '}'
        transforms_per_block = getattr(self.function.meta, 'transforms_per_block', None)
        if transforms_per_block is not None:
            f += ', ' + str(transforms_per_block)
        threads_per_block = getattr(self.function.meta, 'threads_per_block', None)
        if threads_per_block is not None:
            f += ', ' + str(threads_per_block)
        f += ')'
        return f


def generate_cpu_function_pool(functions):
    """Generate function to populate the kernel function pool."""

    function_map_single = Map('function_map_single')
    function_map_double = Map('function_map_double')
    function_map_single_2d = Map('function_map_single_2D')
    function_map_double_2d = Map('function_map_double_2D')
    function_map_single_transpose_tile_aligned = Map('function_map_single_transpose_tile_aligned')
    function_map_double_transpose_tile_aligned = Map('function_map_double_transpose_tile_aligned')
    function_map_single_transpose_diagonal = Map('function_map_single_transpose_diagonal')
    function_map_double_transpose_diagonal = Map('function_map_double_transpose_diagonal')

    pool_map = {
        ('sp', None): function_map_single,
        ('dp', None): function_map_double,
        ('sp', '2D'): function_map_single_2d,
        ('dp', '2D'): function_map_double_2d,
        ('sp', 'tile_aligned'): function_map_single_transpose_tile_aligned,
        ('dp', 'tile_aligned'): function_map_double_transpose_tile_aligned,
        ('sp', 'transpose_diagonal'): function_map_single_transpose_diagonal,
        ('dp', 'transpose_diagonal'): function_map_double_transpose_diagonal,
    }

    populate = StatementList()
    for f in functions:
        length, precision, scheme, pool = f.meta.length, f.meta.precision, f.meta.scheme, f.meta.pool
        if isinstance(length, (int, str)):
            key = Call(name='std::make_pair', arguments=ArgumentList(length, scheme)).inline()
        else:
            key = Call(name='std::make_tuple', arguments=ArgumentList(length[0], length[1], scheme)).inline()
        populate += pool_map[precision, pool].emplace(key, FFTKernel(f))

    return StatementList(
        Include('<iostream>'),
        Include('"../include/function_pool.h"'),
        StatementList(*[f.prototype() for f in functions]),
        Function(name='function_pool::function_pool',
                 value=False,
                 arguments=ArgumentList(),
                 body=populate))


def generate_small_1d_prototypes(precision, transforms):
    """Generate prototypes for 1D small kernels that will be generated by the old generator."""

    data = Variable('data_p', 'const void *')
    back = Variable('back_p', 'void *')
    functions = []

    def add(name, scheme, pool=None):
        functions.append(Function(name=name,
                                  arguments=ArgumentList(data, back),
                                  meta=NS(
                                      length=length,
                                      precision=precision,
                                      scheme=scheme,
                                      pool=pool)))

    for length, scheme in transforms:
        add(f'rocfft_internal_dfn_{precision}_ci_ci_stoc_{length}', scheme)

    return functions


def generate_large_1d_prototypes(precision, transforms):
    """Generate prototypes for 1D large block kernels that will be generated from the old generator."""

    data = Variable('data_p', 'const void *')
    back = Variable('back_p', 'void *')
    functions = []

    def add(name, scheme, pool=None):
        functions.append(Function(name=name,
                                  arguments=ArgumentList(data, back),
                                  meta=NS(
                                      length=length,
                                      precision=precision,
                                      scheme=scheme,
                                      use_3steps_large_twd='false' if (length == 81 or length == 200) else 'true',
                                      pool=pool)))

    for length, scheme in transforms:
        if scheme == 'CS_KERNEL_STOCKHAM_BLOCK_CC':
            add(f'rocfft_internal_dfn_{precision}_ci_ci_sbcc_{length}', 'CS_KERNEL_STOCKHAM_BLOCK_CC')
        elif scheme == 'CS_KERNEL_STOCKHAM_BLOCK_RC':
            add(f'rocfft_internal_dfn_{precision}_op_ci_ci_sbrc_{length}', 'CS_KERNEL_STOCKHAM_BLOCK_RC')
            add(f'rocfft_internal_dfn_{precision}_op_ci_ci_sbrc3d_fft_trans_xy_z_tile_aligned_{length}', 'CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z', 'tile_aligned')
            add(f'rocfft_internal_dfn_{precision}_op_ci_ci_sbrc3d_fft_trans_z_xy_tile_aligned_{length}', 'CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY', 'tile_aligned')
            add(f'rocfft_internal_dfn_{precision}_op_ci_ci_sbrc3d_fft_erc_trans_z_xy_tile_aligned_{length}', 'CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY', 'tile_aligned')
            if length in [128, 256]:
                add(f'rocfft_internal_dfn_{precision}_op_ci_ci_sbrc3d_fft_trans_xy_z_diagonal_{length}', 'CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z', 'transpose_diagonal')
                add(f'rocfft_internal_dfn_{precision}_op_ci_ci_sbrc3d_fft_trans_z_xy_diagonal_{length}', 'CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY', 'transpose_diagonal')
                add(f'rocfft_internal_dfn_{precision}_op_ci_ci_sbrc3d_fft_erc_trans_z_xy_diagonal_{length}', 'CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY', 'transpose_diagonal')

    return functions


def generate_2d_prototypes(precision, transforms):
    """Generate prototypes for 2D kernels that will be generated by the old generator."""

    data = Variable('data_p', 'const void *')
    back = Variable('back_p', 'void *')
    functions = []

    def add(name, scheme, pool=None):
        functions.append(Function(name=name,
                                  arguments=ArgumentList(data, back),
                                  meta=NS(
                                      length=length,
                                      precision=precision,
                                      scheme=scheme,
                                      pool=pool)))

    for length, scheme in transforms:
        add(f'rocfft_internal_dfn_{precision}_ci_ci_2D_{length[0]}_{length[1]}', 'CS_KERNEL_2D_SINGLE', '2D')

    return functions


def list_old_generated_kernels(patterns=None,
                               precisions=None,
                               num_small_kernel_groups=150):
    """Return a list (for CMake) of files created by the (old) generator."""

    if patterns is None:
        patterns = ['all']
    if precisions is None:
        precisions = ['all']

    #
    # all this 'generated_kernels' should go away when the old generator goes away
    #
    generated_kernels = {
        'kernels_launch_basic': [
            'function_pool.cpp',
        ],
        'kernels_launch_small_sp':
          [f'kernel_launch_single_{i}.cpp' for i in range(num_small_kernel_groups)]
          + [f'kernel_launch_single_{i}.cpp.h' for i in range(num_small_kernel_groups)],
        'kernels_launch_small_dp':
          [f'kernel_launch_double_{i}.cpp' for i in range(num_small_kernel_groups)]
          + [f'kernel_launch_double_{i}.cpp.h' for i in range(num_small_kernel_groups)],
        'kernels_launch_large_sp': [
            'kernel_launch_single_large.cpp',
        ],
        'kernels_launch_large_dp': [
            'kernel_launch_double_large.cpp',
        ],
        'kernels_launch_2D_sp': [
            'kernel_launch_single_2D_pow2.cpp',
            'kernel_launch_single_2D_pow3.cpp',
            'kernel_launch_single_2D_pow5.cpp',
            'kernel_launch_single_2D_mix_pow2_3.cpp',
            'kernel_launch_single_2D_mix_pow3_2.cpp',
            'kernel_launch_single_2D_mix_pow3_5.cpp',
            'kernel_launch_single_2D_mix_pow5_3.cpp',
            'kernel_launch_single_2D_mix_pow2_5.cpp',
            'kernel_launch_single_2D_mix_pow5_2.cpp',
        ],
        'kernels_launch_2D_dp': [
            'kernel_launch_double_2D_pow2.cpp',
            'kernel_launch_double_2D_pow3.cpp',
            'kernel_launch_double_2D_pow5.cpp',
            'kernel_launch_double_2D_mix_pow2_3.cpp',
            'kernel_launch_double_2D_mix_pow3_2.cpp',
            'kernel_launch_double_2D_mix_pow3_5.cpp',
            'kernel_launch_double_2D_mix_pow5_3.cpp',
            'kernel_launch_double_2D_mix_pow2_5.cpp',
            'kernel_launch_double_2D_mix_pow5_2.cpp',
        ],
    }
    generated_kernels['kernels_launch_small_all'] = generated_kernels['kernels_launch_small_sp'] + generated_kernels['kernels_launch_small_dp']
    generated_kernels['kernels_launch_large_all'] = generated_kernels['kernels_launch_large_sp'] + generated_kernels['kernels_launch_large_dp']
    generated_kernels['kernels_launch_2D_all']    = generated_kernels['kernels_launch_2D_sp']    + generated_kernels['kernels_launch_2D_dp']
    generated_kernels['kernels_launch_all_sp']    = generated_kernels['kernels_launch_small_sp'] + generated_kernels['kernels_launch_large_sp'] + generated_kernels['kernels_launch_2D_sp']
    generated_kernels['kernels_launch_all_dp']    = generated_kernels['kernels_launch_small_dp'] + generated_kernels['kernels_launch_large_dp'] + generated_kernels['kernels_launch_2D_dp']
    generated_kernels['kernels_launch_all_all']   = generated_kernels['kernels_launch_all_sp']   + generated_kernels['kernels_launch_all_dp']

    gen = generated_kernels['kernels_launch_basic']
    for patt in patterns:
        for prec in precisions:
            gen += generated_kernels[f'kernels_launch_{patt}_{prec}']
    return gen


def list_generated_kernels(kernels):
    """Return list of kernel filenames."""
    return [kernel_file_name(x) for x in kernels]


#
# Main!
#

@name_args(['name', 'ip_fwd', 'ip_inv', 'op_fwd', 'op_inv', 'precision'])
class POWX_SMALL_GENERATOR(BaseNode):
    def __str__(self):
        return f'POWX_SMALL_GENERATOR({cjoin(self.args)});'
    def function(self, meta, precision):
        data = Variable('data_p', 'const void *')
        back = Variable('back_p', 'void *')
        meta = NS(precision=precision, **meta.__dict__)
        return Function(name=self.name,
                        arguments=ArgumentList(data, back),
                        meta=meta)


def kernel_file_name(ns):
    """Given kernel info namespace, return reasonable file name."""
    if hasattr(ns, 'length'):
        length = ns.length
    else:
        length = functools.reduce(lambda a, b: a * b, ns.factors)
    return f'rocfft_len{length}.cpp'


def list_new_kernels():
    """Return list of kernels to generate with the new generator."""

    kernels = [
        NS(length=56,
           threads_per_block=64),
        NS(length=336)
    ]

    return kernels


def generate_new_kernels(kernels):
    """Generate and write kernels from the kernel list.

    Entries in the kernel list are simple namespaces used.  These are
    passed as keyword arguments to the Stockham generator.

    A list of CPU functions is returned.
    """

    fname = Path(__file__).resolve()

    cpu_functions = []
    for kernel in kernels:
        src = StatementList(
            CommentBlock(
                'Stockham kernels generated by:',
                '',
                '    ' + ' '.join(sys.argv),
                '',
                'Generator is: ' + str(fname),
                ''
                'Kernel is: ' + str(kernel)),
            LineBreak(),
            Include('<hip/hip_runtime.h>'),
            Include('"kernel_launch.h"'),
            Include('"rocfft_butterfly_template.h"'),
            LineBreak())

        kdevice, kglobal = stockham.stockham(**kernel.__dict__)
        length = kglobal.meta.length
        forward, inverse = kglobal.name, kglobal.name.replace('forward', 'inverse')
        src += stockham.make_variants(kdevice, kglobal)
        sp = POWX_SMALL_GENERATOR(f'rocfft_internal_dfn_sp_ci_ci_stoc_{length}',
                                  'ip_' + forward, 'ip_' + inverse,
                                  'op_' + forward, 'op_' + inverse, 'float2')
        dp = POWX_SMALL_GENERATOR(f'rocfft_internal_dfn_dp_ci_ci_stoc_{length}',
                                  'ip_' + forward, 'ip_' + inverse,
                                  'op_' + forward, 'op_' + inverse, 'double2')
        src += sp
        src += dp

        cpu_functions += [sp.function(kglobal.meta, 'sp'), dp.function(kglobal.meta, 'dp')]

        format_and_write(kernel_file_name(kernel), src)

    return cpu_functions


def cli():
    """Command line interface..."""
    parser = argparse.ArgumentParser(prog='kernel-generator')
    subparsers = parser.add_subparsers(dest='command')
    parser.add_argument('--groups', type=int, help='Numer of small kernel groups.', default=150)
    parser.add_argument('--pattern', type=str, help='Kernel pattern to generate.', default='all')
    parser.add_argument('--precision', type=str, help='Precision to generate.', default='all')
    parser.add_argument('--manual-small', type=str, help='Small kernel sizes to generate.')
    parser.add_argument('--manual-large', type=str, help='Large kernel sizes to generate.')

    list_parser = subparsers.add_parser('list', help='List kernel files that will be generated.')

    generate_parser = subparsers.add_parser('generate', help='Generate kernels.')
    generate_parser.add_argument('generator', type=str, help='Kernel generator executable.')

    args = parser.parse_args()

    #
    # which kernels to build?
    #
    patterns = args.pattern.split(',')
    large = 'all' in patterns or 'large' in patterns
    small = 'all' in patterns or 'small' in patterns
    dim2  = 'all' in patterns or '2D' in patterns
    pow2  = small or 'pow2' in patterns
    pow3  = small or 'pow3' in patterns
    pow5  = small or 'pow5' in patterns
    pow7  = small or 'pow7' in patterns

    if patterns == ['none']:
        patterns = []
    if args.manual_small:
        patterns += ['small']
    if args.manual_large:
        patterns += ['large']

    replacements = {
        'pow2': 'small',
        'pow3': 'small',
        'pow5': 'small',
        'pow7': 'small',
    }

    patterns = [replacements.get(key, key) for key in patterns if key != 'none']
    if 'all' in patterns:
        patterns = ['all']
    patterns = set(patterns)

    #
    # which precicions to build?
    #
    precisions = args.precision.split(',')

    replacements = {
        'single': 'sp',
        'double': 'dp',
    }

    precisions = [replacements.get(key, key) for key in precisions if key != 'none']
    if 'all' in precisions:
        precisions = ['sp', 'dp']
    precisions = set(precisions)

    new_kernels = list_new_kernels()

    if args.command == 'list':
        scprint(list_old_generated_kernels(patterns=patterns,
                                           precisions=precisions,
                                           num_small_kernel_groups=args.groups)
                + list_generated_kernels(new_kernels))
        return

    manual_small = None
    if args.manual_small:
        manual_small = product(map(int, args.manual_small.split(',')),
                               ['CS_KERNEL_STOCKHAM'])

    manual_large = None
    if args.manual_large:
        manual_large = product(map(int, args.manual_large.split(',')),
                               ['CS_KERNEL_STOCKHAM_BLOCK_CC', 'CS_KERNEL_STOCKHAM_BLOCK_RC'])

    if args.command == 'generate':

        # collection of Functions to generate prototypes for
        psmall, plarge, p2d = {}, {}, {}

        if small or pow2 or pow3 or pow5 or pow7:
            for p in precisions:
                transforms = supported_small_sizes(p, pow2, pow3, pow5, pow7)
                psmall = pmerge(psmall, generate_small_1d_prototypes(p, transforms))

        if manual_small:
            for p in precisions:
                psmall = pmerge(psmall, generate_small_1d_prototypes(p, manual_small))

        if large:
            for p in precisions:
                plarge = pmerge(plarge, generate_large_1d_prototypes(p, supported_large_sizes(p)))

        if manual_large:
            for p in precisions:
                plarge = pmerge(plarge, generate_large_1d_prototypes(p, manual_large))

        if dim2:
            for p in precisions:
                p2d = pmerge(p2d, generate_2d_prototypes(p, supported_2d_sizes(p)))

        # XXX: 2d depends on 1d...

        # hijack a few small kernels...
        pnew = pmerge({}, generate_new_kernels(new_kernels))

        cpu_functions = list(merge(psmall, plarge, p2d, pnew).values())
        format_and_write('function_pool.cpp', generate_cpu_function_pool(cpu_functions))

        old_small_lengths = {f.meta.length for f in psmall.values()}
        new_small_lengths = {f.meta.length for f in pnew.values()}
        gen_small_lengths = old_small_lengths - new_small_lengths
        gen_large_lengths = {f.meta.length for f in plarge.values()}
        if gen_small_lengths:
            subprocess.run([args.generator, '-g', str(args.groups), '-p', args.precision, '-t', 'none', '--manual-small', cjoin(sorted(gen_small_lengths))], check=True)
        if gen_large_lengths:
            subprocess.run([args.generator, '-g', str(args.groups), '-p', args.precision, '-t', 'none', '--manual-large', cjoin(sorted(gen_large_lengths))], check=True)
        if dim2:
            subprocess.run([args.generator, '-g', str(args.groups), '-p', args.precision, '-t', '2D'], check=True)


if __name__ == '__main__':
    cli()
