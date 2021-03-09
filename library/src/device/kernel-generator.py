#!/usr/bin/env python3
'''rocFFT kernel generator.

Currently this acts as a shim between CMake and the C++ kernel generator.

It accept two sub-commands:
1. list - lists files that will be generated
2. generate - pass arguments down to the old generator

'''

import argparse
import os
import sys

from pathlib import Path

top = Path(__file__).resolve().parent

def njoin(xs):
    return '\n'.join([ str(x) for x in xs ])

def cjoin(xs):
    return ';'.join([ str(x) for x in xs ])

def cprint(xs):
    print(cjoin(xs), end='', flush=True)


def list_generated_kernels(pattern=list(['all']), precision=list(['all']), num_small_kernel_groups=150, **kwargs):

    #
    # all this 'generated_kernels' should go away when the old generator goes away
    #

    generated_kernels = {
        'kernels_launch_basic':  [
            'kernel_launch_generator.h',
            'function_pool.cpp.h',
            'function_pool.cpp',
        ],

        'kernels_launch_small_single': \
          [ f'kernel_launch_single_{i}.cpp' for i in range(num_small_kernel_groups) ] + \
          [ f'kernel_launch_single_{i}.cpp.h' for i in range(num_small_kernel_groups) ],

        'kernels_launch_small_double': \
          [ f'kernel_launch_double_{i}.cpp' for i in range(num_small_kernel_groups) ] + \
          [ f'kernel_launch_double_{i}.cpp.h' for i in range(num_small_kernel_groups) ],

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
    for patt in pattern:
        for prec in precision:
            gen += generated_kernels[f'kernels_launch_{patt}_{prec}']
    return gen


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='kernel-generator')
    subparsers = parser.add_subparsers(dest='command')

    list_parser = subparsers.add_parser('list', help='List kernel files that will be generated.')
    list_parser.add_argument('--groups', type=int, help='Numer of small kernel groups.', default=150)
    list_parser.add_argument('--pattern', type=str, help='Kernel pattern to generate.', default='all')
    list_parser.add_argument('--precision', type=str, help='Precision to generate.', default='all')

    generate_parser = subparsers.add_parser('generate', help='Generate kernels.')
    generate_parser.add_argument('generator', type=str, help='Kernel generator executable.')
    generate_parser.add_argument('arguments', type=str, nargs='*', help='Kernel generator arguments.')

    args = parser.parse_args()

    if args.command == 'list':
        cprint(list_generated_kernels(pattern=args.pattern.split(';'),
                                      precision=args.precision.split(';'),
                                      num_small_kernel_groups=args.groups))

    if args.command == 'generate':
        os.execvp(args.generator, ['rocfft-kernel-generator'] + args.arguments)
