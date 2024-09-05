# Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
"""rocfft-bench launch utils."""

import logging
import pathlib
import re
import subprocess
import tempfile
import time


def run(bench,
        length,
        direction=-1,
        real=False,
        inplace=True,
        precision='single',
        nbatch=1,
        ntrial=1,
        device=None,
        libraries=None,
        verbose=False,
        timeout=300,
        sequence=None,
        skiphip=True):
    """Run rocFFT bench and return execution times."""
    cmd = [pathlib.Path(bench).resolve()]

    if libraries is not None:
        for library in libraries:
            cmd += ['--lib', pathlib.Path(library).resolve()]
        if len(libraries) > 1:
            # only use different randomizations if using dyna-bench
            if sequence is not None:
                cmd += ['--sequence', str(sequence)]

    if skiphip:
        cmd += ['--ignore_runtime_failures']
    else:
        cmd += ['--no_ignore_runtime_failures']

    if isinstance(length, int):
        cmd += ['--length', length]
    else:
        cmd += ['--length'] + list(length)

    cmd += ['-N', ntrial]
    cmd += ['-b', nbatch]
    if not inplace:
        cmd += ['-o']
    if precision == 'half':
        cmd += ['--precision', 'half']
    elif precision == 'single':
        cmd += ['--precision', 'single']
    elif precision == 'double':
        cmd += ['--precision', 'double']
    if device is not None:
        cmd += ['--device', device]

    itype, otype = 0, 0
    if real:
        if direction == -1:
            cmd += ['-t', 2, '--itype', 2, '--otype', 3]
        if direction == 1:
            cmd += ['-t', 3, '--itype', 3, '--otype', 2]
    else:
        if direction == -1:
            cmd += ['-t', 0]
        if direction == 1:
            cmd += ['-t', 1]

    cmd = [str(x) for x in cmd]
    logging.info('running: ' + ' '.join(cmd))
    if verbose:
        print('running: ' + ' '.join(cmd))
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")

    time_start = time.time()
    proc = subprocess.Popen(cmd, stdout=fout, stderr=ferr)
    try:
        proc.wait(timeout=None if timeout == 0 else timeout)
    except subprocess.TimeoutExpired:
        logging.info("killed")
        proc.kill()
    time_end = time.time()
    logging.info("elapsed time in seconds: " + str(time_end - time_start))

    fout.seek(0)
    ferr.seek(0)
    cout = fout.read()
    cerr = ferr.read()

    logging.debug(cout)
    logging.debug(cerr)

    tokentoken = "Token: "
    token = ""
    times = []

    soltokenTag = "[SolToken]: "
    soltoken = ""
    matchTag = "[TokenMatch]: "
    match = ""

    for line in cout.splitlines():
        if line.startswith(tokentoken):
            token = line[len(tokentoken):]

    for line in cerr.splitlines():
        if line.startswith(soltokenTag):
            soltoken = line[len(soltokenTag):]
        elif line.startswith(matchTag):
            match = line[len(matchTag):]

    if proc.returncode == 0:
        for m in re.finditer('Execution gpu time: ([ 0-9.]*) ms', cout,
                             re.MULTILINE):
            times.append(list(map(float, m.group(1).split(' '))))
    else:
        logging.info("PROCESS FAILED with return code " + str(proc.returncode))

    if verbose:
        print('finished: ' + ' '.join(cmd))

    if proc.returncode == 0:
        if "SKIPPED" in cout:
            print('s', end='', flush=True)
        elif "HIP_V_THROWERROR" in cout:
            print('h', end='', flush=True)
            # TODO: print hip runtime failed cases?
        else:
            print('.', end='', flush=True)
    else:
        print('x', end='', flush=True)

    success = proc.returncode == 0

    return token, times, success, soltoken, match
