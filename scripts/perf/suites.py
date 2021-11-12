
from itertools import product
from perflib.generators import Problem
from perflib.generators import RadixProblemGenerator

lengths = {

    'md': [
        (100,100,100),
        (160,160,168),
        (160,168,168),
        (160,168,192),
        (160,72,72),
        (160,80,72),
        (160,80,80),
        (168,168,192),
        (168,192,192),
        (168,80,80),
        (192,192,192),
        (192,192,200),
        (192,200,200),
        (192,84,84),
        (192,96,84),
        (192,96,96),
        (200,100,96),
        (200,200,200),
        (200,96,96),
        (208,100,100),
        (216,104,100),
        (216,104,104),
        (224,104,104),
        (224,108,104),
        (224,108,108),
        (240,108,108),
        (240,112,108),
        (240,112,112),
        (60,60,60),
        (64,64,52),
        (64,64,64),
        (72,72,52),
        (72,72,72),
        (80,80,80),
        (84,84,72),
        (96,96,96),

        (108,108,80),
        (216,216,216),

        (128,128,256),
        (240,224,224),
        (64,64,64),
        (80,84,14),
        (80,84,144),

        (25,20,20),
        (42,32,32),
        (75,55,55)
    ],

    'generated': [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18,
                   20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 36, 40, 42, 44, 45,
                   48, 49, 50, 52, 54, 56, 60, 64, 72, 75, 80, 81, 84, 88, 90,
                   96, 100, 104, 108, 112, 120, 121, 125, 128, 135, 144, 150,
                   160, 162, 168, 169, 176, 180, 192, 200, 208, 216, 224, 225,
                   240, 243, 250, 256, 270, 288, 300, 320, 324, 336, 343, 360,
                   375, 384, 400, 405, 432, 450, 480, 486, 500, 512, 540, 576,
                   600, 625, 640, 648, 675, 720, 729, 750, 768, 800, 810, 864,
                   900, 960, 972, 1000, 1024, 1080, 1125, 1152, 1200, 1215,
                   1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536, 1600, 1620,
                   1728, 1800, 1875, 1920, 1944, 2000, 2025, 2048, 2160, 2187,
                   2250, 2304, 2400, 2430, 2500, 2560, 2592, 2700, 2880, 2916,
                   3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750,
                   3840, 3888, 4000, 4050, 4096],
}


def mktag(tag, dimension, precision, direction, inplace, real):
    t = [tag,
         str(dimension) + 'D',
         precision,
         {-1: 'forward', 1: 'backward'}[direction],
         {True: 'real', False: 'complex'}[real],
         {True: 'in-place', False: 'out-of-place'}[inplace]]
    return "_".join(t)


def md():
    """Molecular dynamics suite."""

    precisions = ['single', 'double']
    directions = [-1, 1]
    inplaces   = [True,False]
    reals      = [True,False]

    for precision, direction, inplace, real in product(precisions, directions, inplaces, reals):
        for length in lengths['md']:
            nbatch = 10
            yield Problem(length,
                          tag=mktag("md", len(length), precision, direction, inplace, real),
                          nbatch=nbatch,
                          direction=direction,
                          inplace=inplace,
                          real=real,
                          precision=precision)

def qa():
    """AMD QA suite."""

    for length1 in [8192, 10752, 18816, 21504, 32256, 43008, 16384, 16807]:
        for direction in [-1, 1]:
            yield Problem([length1],
                          tag=mktag("qa1", 1, 'double', direction, False, False),
                          nbatch=10000,
                          direction=direction,
                          inplace=False,
                          real=False,
                          precision='double')

    yield Problem([10000],
                  tag=mktag('qa10k', 1, 'double', 1, False, False),
                  nbatch=10000,
                  direction=1,
                  inplace=False,
                  real=False,
                  precision='double')

    yield Problem([(336,336,56)],
                  tag=mktag('qa3', 3, 'double', -1, False, False),
                  nbatch=1,
                  direction=-1,
                  inplace=False,
                  real=False,
                  precision='double')

    for length3 in lengths['md']:
        for direction in [-1, 1]:
            yield Problem(length3,
                          tag=mktag('qa3md', 3, 'single', direction, inplace, False, True),
                          nbatch=1,
                          direction=direction,
                          inplace=False,
                          real=True,
                          precision='single')


def generated1d():
    """Explicitly generated 1D lengths."""

    precisions = ['single', 'double']
    directions = [-1, 1]
    inplaces   = [True,False]
    reals      = [True,False]

    for precision, direction, inplace, real in product(precisions, directions, inplaces, reals):
        for length in lengths['generated']:
            yield Problem([length],
                          tag=mktag("generated1d", 1, precision, direction, inplace, real),
                          nbatch=1000,
                          direction=direction,
                          inplace=inplace,
                          real=real,
                          precision=precision)

def generated2d():
    """Explicitly generated 2D lengths."""

    lengths2d  = list(filter(lambda x: x <= 1024, lengths['generated']))
    precisions = ['single', 'double']
    directions = [-1, 1]
    inplaces   = [True,False]
    reals      = [True,False]

    for precision, direction, inplace, real in product(precisions, directions, inplaces, reals):
        for length in lengths2d:
            yield Problem([length,length],
                          tag=mktag("generated2d", 2, precision, direction, inplace, real),
                          nbatch=1000,
                          direction=direction,
                          inplace=inplace,
                          real=real,
                          precision=precision)

def benchmarks():
    """Benchmarks: XXX"""

    pow2 = [2**k for k in range(30)]
    pow3 = [3**k for k in range(19)]
    minmax = {
        1: (256, 536870912),
        2: (64, 32768),
        3: (16, 1024),
    }

    lengths    = sorted(pow2 + pow3)
    dimensions = [1, 2, 3]
    precisions = ['single', 'double']
    directions = [-1, 1]
    inplaces   = [True,False]
    reals      = [True,False]

    for dimension, precision, direction, inplace, real in product(dimensions, precisions, directions, inplaces, reals):
        min1, max1 = minmax[dimension]
        for length in lengths:
            if min1 <= length <= max1:
                yield Problem((3*[length])[:dimension],
                              tag=mktag('benchmark', dimension, precision, direction, inplace, real),
                              nbatch=1,
                              direction=direction,
                              inplace=inplace,
                              real=real,
                              precision=precision)
