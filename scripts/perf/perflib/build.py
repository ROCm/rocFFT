"""Utilities to build rocFFT."""

import subprocess
import logging
from . import git
from pathlib import Path
from .utils import sjoin
from shutil import which

def local(cmd, echo=True, **kwargs):
    """Run `cmd` using the shell.

    Keyword arguments are passed down to `subprocess.run`.
    """
    if echo:
        print('local: ' + cmd)
    logging.info('local: ' + cmd)
    return subprocess.run(cmd, shell=True, **kwargs)


def local_amdgpu_target():
    try:
        for line in subprocess.Popen(args=["rocminfo"], stdout=subprocess.PIPE).stdout.readlines():
            if b'amdgcn-amd-amdhsa--' in line:
                return line.split(b'--')[1].strip().decode('utf-8')
    except:
        pass
    return ''


def build_rocfft(commit, dest=None, repo='git@github.com:ROCmSoftwarePlatform/rocFFT-internal.git', ccache=False):
    """Build public rocFFT (at specified git `commit`) and install into `dest`."""

    top = Path('.').resolve() / ('rocFFT-' + commit)

    if not top.exists():
        git.clone(repo, top)
    git.checkout(top, commit)

    if git.is_dirty(top):
        print(f'working directory {top} is dirty!')
        raise SystemExit

    build = top / 'build'
    build.mkdir(exist_ok=True)
    defs = ['-DCMAKE_CXX_COMPILER=hipcc',
            '-DBUILD_CLIENTS_RIDER=ON',
            '-DROCFFT_CALLBACKS_ENABLED=OFF',
            '-DSINGLELIB=ON',
            '-DAMDGPU_TARGETS=' + local_amdgpu_target()]
    if dest:
        defs += [f'-DCMAKE_INSTALL_PREFIX={dest}']
    if ccache:
        defs += ['-DCMAKE_CXX_COMPILER_LAUNCHER=ccache']

    use_ninja = which('ninja') is not None

    if use_ninja:
        defs += ['-G', 'Ninja']

    local(f'cmake {sjoin(defs)} ..', cwd=build, check=True)
    if use_ninja:
        local('ninja', cwd=build, check=True)
    else:
        local('make -j $(nproc)', cwd=build, check=True)

    if dest:
        if use_ninja:
            local('ninja install', cwd=build, check=True)
        else:
            local('make install', cwd=build, check=True)
        local(f'cp {build}/clients/staging/* {dest}')
