#!/usr/bin/env python3
"""rocFFT kernel generator.

Currently this acts as a shim between CMake and the C++ kernel generator.

It accept two sub-commands:
1. list - lists files that will be generated
2. generate - pass arguments down to the old generator

Note that 'small' kernels don't decompose their lengths.

"""

import argparse
import subprocess

from types import SimpleNamespace as NS
from itertools import product

from generator import *


#
# CMake helpers
#


def scjoin(xs):
    return ';'.join(str(x) for x in xs)


def scprint(xs):
    print(scjoin(xs), end='', flush=True)


#
# Supported kernel sizes
#

def supported_small_sizes(precision):
    """Return list of 1D small kernels."""

    upper_bound = {
        'single': 4096,
        'double': 4096,         # of course this isn't 2048... not sure why (double len 1594323 will fail)
    }

    powers = {
        5: [5**k for k in range(6)],
        3: [3**k for k in range(8)],
        2: [2**k for k in range(13)],
    }

    lengths = [p2 * p3 * p5 for p2, p3, p5 in product(powers[2], powers[3], powers[5])]

    # common radix 7, 11, and 13
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
    bytes_per_element = {'single': 8, 'double': 16}[precision]

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


def generate_cpu_header(functions):
    """Generate C prototypes of 'functions'."""

    prototypes = StatementList()
    for f in functions:
        prototypes += f.prototype()

    hdr = StatementList()
    hdr += Pragma('once')
    hdr += ExternC(prototypes)

    return hdr


def generate_cpu_function_pool(functions):
    """Generate function to populate the kernel function pool."""

    function_map_single = Variable('function_map_single', array=True)
    function_map_double = Variable('function_map_double', array=True)
    function_map_single_2d = Variable('function_map_single_2D', array=True)
    function_map_double_2d = Variable('function_map_double_2D', array=True)
    function_map_single_transpose_tile_aligned = Variable('function_map_single_transpose_tile_aligned', array=True)
    function_map_double_transpose_tile_aligned = Variable('function_map_double_transpose_tile_aligned', array=True)
    function_map_single_transpose_diagonal = Variable('function_map_single_transpose_diagonal', array=True)
    function_map_double_transpose_diagonal = Variable('function_map_double_transpose_diagonal', array=True)

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

    addpool = StatementList()
    for f in functions:
        length, precision, scheme, pool = f.meta.length, f.meta.precision, f.meta.scheme, f.meta.pool
        if isinstance(length, int):
            key = Call(name='std::make_pair', arguments=ArgumentList(length, scheme)).inline()
        else:
            key = Call(name='std::make_tuple', arguments=ArgumentList(length[0], length[1], scheme)).inline()
        addpool += Assign(pool_map[precision, pool][key], f.address())

    hdr = StatementList()
    hdr += Include('<iostream>')
    hdr += Include('"../include/function_pool.h"')
    hdr += Include('"kernel_launch_generator.h"')
    hdr += Function(name='function_pool::function_pool',
                    value=False,
                    arguments=ArgumentList(),
                    body=addpool)

    return hdr


def old_generate_small_1d_prototypes(precision, transforms):
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


def old_generate_large_1d_prototypes(precision, transforms):
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


def old_generate_2d_prototypes(precision, transforms):
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



def list_generated_kernels(patterns=None,
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
            'kernel_launch_generator.h',
            'function_pool.cpp.h',
            'function_pool.cpp',
        ],
        'kernels_launch_small_single':
          [f'kernel_launch_single_{i}.cpp' for i in range(num_small_kernel_groups)]
          + [f'kernel_launch_single_{i}.cpp.h' for i in range(num_small_kernel_groups)],
        'kernels_launch_small_double':
          [f'kernel_launch_double_{i}.cpp' for i in range(num_small_kernel_groups)]
          + [f'kernel_launch_double_{i}.cpp.h' for i in range(num_small_kernel_groups)],
        'kernels_launch_large_single': [
            'kernel_launch_single_large.cpp',
        ],
        'kernels_launch_large_double': [
            'kernel_launch_double_large.cpp',
        ],
        'kernels_launch_2D_single': [
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
        'kernels_launch_2D_double': [
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
    generated_kernels['kernels_launch_small_all'] = generated_kernels['kernels_launch_small_single'] + generated_kernels['kernels_launch_small_double']
    generated_kernels['kernels_launch_large_all'] = generated_kernels['kernels_launch_large_single'] + generated_kernels['kernels_launch_large_double']
    generated_kernels['kernels_launch_2D_all'] = generated_kernels['kernels_launch_2D_single'] + generated_kernels['kernels_launch_2D_double']
    generated_kernels['kernels_launch_all_single'] = generated_kernels['kernels_launch_small_single'] + generated_kernels['kernels_launch_large_single'] + generated_kernels['kernels_launch_2D_single']
    generated_kernels['kernels_launch_all_double'] = generated_kernels['kernels_launch_small_double'] + generated_kernels['kernels_launch_large_double'] + generated_kernels['kernels_launch_2D_double']
    generated_kernels['kernels_launch_all_all'] = generated_kernels['kernels_launch_all_single'] + generated_kernels['kernels_launch_all_double']

    gen = generated_kernels['kernels_launch_basic']
    for patt in patterns:
        for prec in precisions:
            gen += generated_kernels[f'kernels_launch_{patt}_{prec}']
    return gen


#
# Main!
#

if __name__ == '__main__':

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

    patterns = args.pattern.split(',')
    precisions = args.precision.split(',')
    if patterns == ['none']:
        patterns = []
    if args.manual_small:
        patterns += ['small']
    if args.manual_large:
        patterns += ['large']

    manual_small = None
    if args.manual_small:
        manual_small = product(args.manual_small.split(','), ['CS_KERNEL_STOCKHAM'])

    manual_large = None
    if args.manual_large:
        manual_large = product(args.manual_large.split(','),
                               ['CS_KERNEL_STOCKHAM_BLOCK_CC', 'CS_KERNEL_STOCKHAM_BLOCK_RC'])

    single = 'all' in precisions or 'single' in precisions
    double = 'all' in precisions or 'double' in precisions

    large = 'all' in patterns or 'large' in patterns
    small = 'all' in patterns or 'small' in patterns

    if args.command == 'list':
        scprint(list_generated_kernels(patterns=patterns,
                                       precisions=precisions,
                                       num_small_kernel_groups=args.groups))

    if args.command == 'generate':

        cpu_functions = []

        old_args = ['-t', args.pattern, '-p', args.precision, '-g', str(args.groups)]

        if manual_small:
            old_args += ['--manual-small', args.manual_small]
            if single:
                cpu_functions += old_generate_small_1d_prototypes('sp', manual_small)
            if double:
                cpu_functions += old_generate_small_1d_prototypes('dp', manual_small)
        elif small:
            if single:
                cpu_functions += old_generate_small_1d_prototypes('sp', supported_small_sizes('single'))
            if double:
                cpu_functions += old_generate_small_1d_prototypes('dp', supported_small_sizes('double'))

        if manual_large:
            old_args += ['--manual-large', args.manual_large]
            if single:
                cpu_functions += old_generate_large_1d_prototypes('sp', manual_large)
            if double:
                cpu_functions += old_generate_large_1d_prototypes('dp', manual_large)
        elif large:
            if single:
                cpu_functions += old_generate_large_1d_prototypes('sp', supported_large_sizes('single'))
            if double:
                cpu_functions += old_generate_large_1d_prototypes('dp', supported_large_sizes('double'))

        if single:
            cpu_functions += old_generate_2d_prototypes('sp', supported_2d_sizes('single'))
        if double:
            cpu_functions += old_generate_2d_prototypes('dp', supported_2d_sizes('double'))

        # XXX: 2d depends on 1d...

        format_and_write('kernel_launch_generator.h', generate_cpu_header(cpu_functions))
        format_and_write('function_pool.cpp.h', generate_cpu_function_pool(cpu_functions))
        format_and_write('function_pool.cpp', Include('"function_pool.cpp.h"'))

        subprocess.run([args.generator] + old_args)
