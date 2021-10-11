#!/usr/bin/env python3
import sys
import os
import argparse
import hashlib

def filename_to_cpp_ident(filename):
    base = os.path.basename(filename)
    return base.replace('-', '_').replace('.', '_')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Write embedded C++ generator file")
    parser.add_argument('--embed', metavar='file', type=str, nargs='+', required=True,
                        help='files to embed into the header')
    parser.add_argument('--logic', metavar='file', type=str, nargs='+', required=True,
                        help='additional files that make up generator logic')
    parser.add_argument('--output', metavar='file', type=str, required=True,
                        help='output file')
    args = parser.parse_args()

    output = args.output

    outfile = open(output, 'w')

    # embed files as strings
    outfile.write("#pragma once\n")
    for input in args.embed:
        ident = filename_to_cpp_ident(input)
        outfile.write(f"static constexpr auto {ident} {{\n")
        outfile.write('R"_PY_EMBED_(\n')
        with open(input, 'r') as f:
            for line in f:
                outfile.write(line)
        outfile.write(')_PY_EMBED_"};\n')

    # hash input files, write sum
    h = hashlib.sha256()
    for input in args.embed + args.logic:
        with open(input, 'rb') as f:
            h.update(f.read())
    outfile.write('static const char* generator_sum = "')
    for b in h.digest():
        outfile.write('\\x{:02x}'.format(b))
    outfile.write('";\n')
    outfile.write('static const size_t generator_sum_bytes = {};\n'.format(h.digest_size))
