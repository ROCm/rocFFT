"""Rider launch utils."""

import logging
import pathlib
import re
import subprocess
import tempfile
import time


def run(rider, length, direction=-1, real=False, inplace=True,
        precision='single', nbatch=1, ntrial=1, device=None,
        libraries=None, verbose=False, timeout=300):
    """Run rocFFT rider and return execution times."""
    cmd = [pathlib.Path(rider).resolve()]

    if isinstance(length, int):
        cmd += ['--length', length]
    else:
        cmd += ['--length'] + list(length)

    if libraries is not None:
        for library in libraries:
            cmd += ['--lib', pathlib.Path(library).resolve()]

    cmd += ['-N', ntrial]
    cmd += ['-b', nbatch]
    if not inplace:
        cmd += ['-o']
    if precision == 'double':
        cmd += ['--double']
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
    else:
        print('.', end='', flush=True)
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")

    time_start = time.time()
    proc = subprocess.Popen(cmd,
                            stdout=fout,
                            stderr=ferr)
    try:
        proc.wait(timeout=None if timeout == 0 else timeout)
    except subprocess.TimeoutExpired:
        logging.info("killed")
        proc.kill()
    time_end = time.time()
    logging.info("elapsed time in seconds: " + str(time_end - time_start))
    
    
    cout = fout.read()
    cerr = ferr.read()

    logging.debug(cout)

    times = []
    if proc.returncode == 0:
        for m in re.finditer('Execution gpu time: ([ 0-9.]*) ms', cout, re.MULTILINE):
            times.append(list(map(float, m.group(1).split(' '))))

    return times
