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
"""A few small utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from functools import reduce

import sys

#
# Join shortcuts
#


def join(sep, s):
    """Return 's' joined with 'sep'.  Coerces to str."""
    return sep.join(str(x) for x in list(s))


def sjoin(s):
    """Return 's' joined with spaces."""
    return join(' ', [str(x) for x in s])


def njoin(s):
    """Return 's' joined with newlines."""
    return join('\n', s)


def cjoin(s):
    """Return 's' joined with commas."""
    return join(',', s)


def tjoin(s):
    """Return 's' joined with tabs."""
    return join('\t', s)


#
# Misc
#


def shape(n, nbatch):
    """Return NumPy shape."""
    if isinstance(n, (list, tuple)):
        return [nbatch] + list(n)
    return [nbatch, n]


def product(xs):
    """Return product of factors."""
    return reduce(lambda x, y: x * y, xs, 1)


def flatten(xs):
    """Flatten list of lists to a list."""
    return sum(xs, [])


def write_tsv(path, records, meta={}, overwrite=False):
    """Write tab separated file."""
    path = Path(path)
    dat = []
    with open(path, 'a') as f:
        if overwrite:
            f.truncate(0)
        if f.tell() == 0:
            if meta is not None:
                for k, v in meta.items():
                    dat.append(f'# {k}: {v}')
        dat += [tjoin([str(x) for x in r]) for r in records]
        f.write(njoin(dat))
        f.write('\n')


def write_csv(path, records, meta={}, overwrite=False):
    """Write commas separated file."""
    path = Path(path)
    dat = []
    with open(path, 'a') as f:
        if overwrite:
            f.truncate(0)
            if meta is not None:
                for k, v in meta.items():
                    dat.append(f'# {k}: {v}')
        dat += [cjoin([str(x) for x in r]) for r in records]
        f.write(njoin(dat))
        f.write('\n')


#
# DAT files
#


@dataclass
class Sample:
    """Dyna-bench/bench timing sample: list of times for a given token.

    This corresponds to a single line of a dat file.
    """

    token: str
    times: List[float]
    label: str = None

    def __post_init__(self):
        self.label = self.token


@dataclass
class DAT:
    """Dyna-bench/bench DAT.

    This corresponds to a single .dat file.
    """

    tag: str
    path: Path
    samples: Dict[str, Sample]
    meta: Dict[str, str]

    def get_samples(self):
        keys = self.samples.keys()
        for key in keys:
            yield key, self.samples[key]

    def print(self):
        print("tag:", self.tag)
        print("path:", self.path)
        print("meta:", self.meta)
        print("samples:", self.samples)


@dataclass
class Run:
    """Dyna-bench/bench runs.

    This corresponds to a directory of .dat files.
    """

    title: str
    path: Path
    dats: Dict[Path, DAT]


def write_dat(fname, token, seconds, meta={}):
    """Append record to dyna-bench/bench .dat file."""
    record = [token, len(seconds)] + seconds
    write_tsv(fname, [record], meta=meta, overwrite=False)


def parse_token(token):
    words = token.split("_")

    precision = None
    length = []
    transform_type = None
    batch = None
    placeness = None

    if words[0] not in {"complex", "real"}:
        print("Error parsing token:", token)
        sys.exit(1)
    if words[1] not in {"forward", "inverse"}:
        print("Error parsing token:", token)
        sys.exit(1)
    transform_type = ("forward" if words[1] == "forward" else
                      "backward") + "_" + words[0]

    lendidx = -1
    for idx in range(len(words)):
        if words[idx] == "len":
            lenidx = idx
            break
    for idx in range(lenidx + 1, len(words)):
        if words[idx].isnumeric():
            length.append(int(words[idx]))
        else:
            # Now we have the precision and placeness
            precision = words[idx]
            placeness = "out-of-place" if words[idx +
                                                1] == "op" else "in-place"
            break

    batchidx = -1
    for idx in range(len(words)):
        if words[idx] == "batch":
            batchidx = idx
            break
    batch = []
    for idx in range(batchidx + 1, len(words)):
        if words[idx].isnumeric():
            batch.append(int(words[idx]))
        else:
            break

    return transform_type, placeness, length, batch, precision


def read_dat(fname):
    """Read dyna-bench/bench .dat file."""
    path = Path(fname)
    records, meta = {}, {}
    for line in path.read_text().splitlines():
        if line.startswith('# '):
            k, v = [x.strip() for x in line[2:].split(':', 1)]
            meta[k] = v
            continue
        words = line.split("\t")
        token = words[0]
        times = list(map(float, words[2:]))
        records[token] = Sample(token, times)
    tag = meta['title'].replace(' ', '_')
    return DAT(tag, path, records, meta)


def read_run(dname, verbose=False):
    """Read all .dat files in a directory."""
    path = Path(dname)
    if verbose:
        print("reading", path)
    dats = {}
    for dat in list_runs(dname):
        dats[dat.stem] = read_dat(dat)
    return Run(path.stem, path, dats)


def list_runs(dname):
    """List all .dat files in a directory."""
    path = Path(dname)
    return sorted(list(path.glob('*.dat')))


def read_runs(dnames, verbose=False):
    """Read all .dat files in directories."""
    return [read_run(dname, verbose) for dname in dnames]


def get_post_processed(dname, docdir, outdirs):
    """Return file names of post-processed performance data.

    The 'primary' files contain median confidence intervals for each
    DAT file.

    The 'secondary' files contain XXX.
    """
    primary = []
    for outdir in outdirs:
        path = (Path(outdir) / dname).with_suffix('.mdat')
        if path.exists():
            primary.append(path)

    secondary = []
    for outdir in outdirs[1:]:
        path = (docdir / (str(outdir.name) + "-over-" + str(outdirs[0].name) +
                          "-" + dname)).with_suffix('.sdat')
        if path.exists():
            secondary.append(path)

    return primary, secondary


def by_dat(runs):
    r = {}
    for dat in runs[0].dats.values():
        dstem = dat.path.stem
        r[dstem] = {
            run.path: run.dats[dstem]
            for run in runs if dstem in run.dats
        }
    return r


def to_data_frames(primaries, secondaries):
    import pandas
    data_frames = []
    for primary in primaries:
        df = pandas.read_csv(primary, delimiter='\t', comment='#')
        data_frames.append(df)

    for i, secondary in enumerate(secondaries):
        df = pandas.read_csv(secondary, delimiter='\t', comment='#')
        data_frames[i + 1] = data_frames[i + 1].merge(df,
                                                      how='left',
                                                      on='token',
                                                      suffixes=('', '_y'))

    return data_frames


def write_pts_dat(fname, records, meta={}):
    """Write data to *.ptsdat"""
    write_csv(fname, records, meta=meta, overwrite=True)
