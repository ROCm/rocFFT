"""A few small utilities."""

import pandas

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from functools import reduce


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
            if meta is not None:
                for k, v in meta.items():
                    dat.append(f'# {k}: {v}')
        dat += [tjoin([str(x) for x in r]) for r in records]
        f.write(njoin(dat))


#
# DAT files
#

@dataclass
class Sample:
    """Dyna-rider/rider timing sample: list of times for a given length+batch.

    This corresponds to a single line of a dat file.
    """

    lengths: List[int]
    nbatch: int
    times: List[float]
    label: str = None

    def __post_init__(self):
        self.label = 'x'.join(map(str, self.lengths)) + 'b' + str(self.nbatch)


@dataclass
class DAT:
    """Dyna-rider/rider DAT.

    This corresponds to a single .dat file.
    """

    tag: str
    path: Path
    samples: Dict[Tuple, Sample]
    meta: Dict[str, str]

    def sorted_samples(self):
        keys = sorted(self.samples.keys(), key=lambda x: product(x))
        for key in keys:
            yield key, product(key), self.samples[key]


@dataclass
class Run:
    """Dyna-rider/rider runs.

    This corresponds to a directory of .dat files.
    """

    title: str
    path: Path
    dats: Dict[Path, DAT]


def write_dat(fname, length, nbatch, seconds, meta={}):
    """Append record to dyna-rider/rider .dat file."""
    if isinstance(length, int):
        length = [length]
    record = [len(length)] + list(length) + [nbatch, len(seconds)] + seconds
    write_tsv(fname, [record], meta=meta, overwrite=False)


def read_dat(fname):
    """Read dyna-rider/rider .dat file."""
    path = Path(fname)
    records, meta = {}, {}
    for line in path.read_text().splitlines():
        if line.startswith('# '):
            k, v = [x.strip() for x in line[2:].split(':', 1)]
            meta[k] = v
            continue
        words   = line.split("\t")
        dim     = int(words[0])
        lengths = tuple(map(int, words[1:dim + 1]))
        nbatch  = int(words[dim + 1])
        times   = list(map(float, words[dim + 3:]))
        records[lengths] = Sample(list(lengths), nbatch, times)
    tag = meta['title'].replace(' ', '_')
    return DAT(tag, path, records, meta)


def read_run(dname):
    """Read all .dat files in a directory."""
    path = Path(dname)
    dats = {}
    for dat in sorted(path.glob('**/*.dat')):
        dats[dat.stem] = read_dat(dat)
    return Run(path.stem, path, dats)


def list_run(dname):
    """List all .dat files in a directory."""
    path = Path(dname)
    return sorted(list(path.glob('*.dat')))


def read_runs(dnames):
    """Read all .dat files in directories."""
    return [read_run(dname) for dname in dnames]


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
        path = (docdir / (str(outdir.name) + "-over-" + str(outdirs[0].name) + "-" + dname)).with_suffix('.sdat')
        if path.exists():
            secondary.append(path)

    return primary, secondary


def by_dat(runs):
    r = {}
    for dat in runs[0].dats.values():
        dstem = dat.path.stem
        r[dstem] = {
            run.path: run.dats[dstem] for run in runs if dstem in run.dats
        }
    return r


def to_data_frames(primaries, secondaries):

    data_frames = []
    for primary in primaries:
        df = pandas.read_csv(primary, delimiter='\t', comment='#')
        data_frames.append(df)

    for i, secondary in enumerate(secondaries):
        df = pandas.read_csv(secondary, delimiter='\t', comment='#')
        data_frames[i+1] = data_frames[i+1].merge(df, how='left', on='length', suffixes=('', '_y'))

    return data_frames
