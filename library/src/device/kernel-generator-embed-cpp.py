#!/usr/bin/env python3
import sys
import os

def filename_to_cpp_ident(filename):
    base = os.path.basename(filename)
    return base.replace('-', '_').replace('.', '_')

if __name__ == '__main__':
    output = sys.argv[-1]
    inputs = sys.argv[1:-1]

    outfile = open(output, 'w')

    outfile.write("#pragma once\n")
    for input in inputs:
        ident = filename_to_cpp_ident(input)
        outfile.write(f"static constexpr auto {ident} {{\n")
        outfile.write('R"_PY_EMBED_(\n')
        with open(input, 'r') as f:
            for line in f:
                outfile.write(line)
        outfile.write(')_PY_EMBED_"};\n')

