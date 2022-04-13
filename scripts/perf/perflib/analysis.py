"""Performance analysis routines."""

import random

import numpy as np
import statistics

from perflib.utils import Run
from dataclasses import dataclass
from typing import List


def confidence_interval(vals, alpha=0.95, nboot=2000):
    """Compute the alpha-confidence interval for the given values using boot-strap resampling."""
    medians = []
    for iboot in range(nboot):
        resample = []
        for i in range(len(vals)):
            resample.append(vals[random.randrange(len(vals))])
        medians.append(np.median(resample))
    medians = sorted(medians)
    low = medians[int(np.floor(nboot * 0.5 * (1.0 - alpha)))]
    high = medians[int(np.ceil(nboot * (1.0 - 0.5 * (1.0 - alpha))))]
    return low, high


def ratio_confidence_interval(Avals, Bvals, alpha=0.95, nboot=2000):
    """Compute the alpha-confidence interval for the ratio of the given sets of values using boot-strap resampling."""
    ratios = []
    for i in range(nboot):
        ratios.append(Avals[random.randrange(len(Avals))] / Bvals[random.randrange(len(Bvals))])
    ratios = sorted(ratios)
    low = ratios[int(np.floor(len(ratios) * 0.5 * (1.0 - alpha)))]
    high = ratios[int(np.ceil(len(ratios) * (1.0 - 0.5 * (1.0 - alpha))))]
    return low, high


@dataclass
class MoodsResult:
    pval: float
    medians: List[float]


def moods(reference: Run, others: List[Run]):
    """Perform Moods analysis..."""
    import scipy.stats
    pvals = {}
    for rname, rdat in reference.dats.items():
        for other in others:
            odat = other.dats[rname]
            for length in rdat.samples.keys():
                s1 = rdat.samples[length].times
                s2 = odat.samples[length].times
                m1 = statistics.median(s1)
                m2 = statistics.median(s2)
                _, p, _, _ = scipy.stats.median_test(s1, s2)
                pvals[other.path.name, rname, length] = MoodsResult(p, [m1, m2])
    return pvals
