
from itertools import product
from perflib.generators import Problem
from perflib.generators import RadixProblemGenerator


all_precisions = ['single', 'double']
all_directions = [-1, 1]
all_inplaces   = [True,False]
all_reals      = [True,False]

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
        (75,55,55),
    ],

    'misc3d': [
        (256, 256, 256),
        (336, 336, 56),
    ],

    'misc2d': [
        (256, 256),
        (56, 336),
        (4096, 4096),
        (336, 18816),
    ],

    'simpleL1D': [
        6561,
        8192,
        10000,
        16384,
        32768,
        40000,
        65536,
    ],

    'large1d': [
        8192,
        10000,
        10752,
        15625,
        16384,
        16807,
        18816,
        19683,
        21504,
        32256,
        43008,
    ],

    'mixed': [
        225, 240, 300, 486, 600, 900, 958, 1014, 1139,
        1250, 1427, 1463, 1480, 1500, 1568, 1608, 1616, 1638, 1656,
        1689, 1696, 1708, 1727, 1744, 1752, 1755, 1787, 1789, 1828,
        1833, 1845, 1860, 1865, 1875, 1892, 1897, 1899, 1900, 1903,
        1905, 1912, 1933, 1938, 1951, 1952, 1954, 1956, 1961, 1964,
        1976, 1997, 2004, 2005, 2006, 2012, 2016, 2028, 2033, 2034,
        2038, 2069, 2100, 2113, 2116, 2123, 2136, 2152, 2160, 2167,
        2181, 2182, 2187, 2205, 2208, 2242, 2250, 2251, 2288, 2306,
        2342, 2347, 2352, 2355, 2359, 2365, 2367, 2383, 2385, 2387,
        2389, 2429, 2439, 2445, 2448, 2462, 2467, 2474, 2478, 2484,
        2486, 2496, 2500, 2503, 2519, 2525, 2526, 2533, 2537, 2556,
        2558, 2559, 2566, 2574, 2576, 2594, 2604, 2607, 2608, 2612,
        2613, 2618, 2632, 2635, 2636, 2641, 2652, 2654, 2657, 2661,
        2663, 2678, 2688, 2690, 2723, 2724, 2728, 2729, 2733, 2745,
        2755, 2760, 2772, 2773, 2780, 2786, 2789, 2790, 2805, 2807,
        2808, 2812, 2815, 2816, 2820, 2826, 2830, 2834, 2841, 2847,
        2848, 2850, 2852, 2853, 2872, 2877, 2882, 2883, 2886, 2887,
        2892, 2893, 2917, 2922, 2924, 2926, 2928, 2929, 2932, 2933,
        2934, 2938, 2951, 2960, 2970, 2979, 2990, 2994, 2998, 2999,
        3000, 3001, 3003, 3004, 3008, 3034, 3035, 3039, 3040, 3042,
        3048, 3052, 3055, 3060, 3065, 4000, 12000, 24000,
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

    'qa1d10b': [
        16777216,
        14348907,
        9765625,
    ],

    'qa2d10b': [
        (4096, 4096),
        (6561, 6561),
        (3125, 3125),
    ],

    'qa3d10b': [
        (256, 256, 256),
        (243, 243, 243),
        (125, 125, 125),
    ],

    'qaReal3d10b': [
        (100, 100, 100),
        (200, 200, 200),
        (192, 192, 192),
    ],
}


def mktag(tag, dimension, precision, direction, inplace, real):
    t = [tag,
         str(dimension) + 'D',
         precision,
         {-1: 'forward', 1: 'backward'}[direction],
         {True: 'real', False: 'complex'}[real],
         {True: 'in-place', False: 'out-of-place'}[inplace]]
    return "_".join(t)


# yield problem sizes with default precision, direction, etc
def default_length_params(tag, lengths, nbatch, precisions=all_precisions, \
    directions=all_directions, inplaces=all_inplaces, reals=all_reals):

    for precision, direction, inplace, real in product(precisions, directions, inplaces, reals):
        for length in lengths:
            length = (length,) if isinstance(length,int) else length
            yield Problem(length,
                          tag=mktag(tag, len(length), precision, direction, inplace, real),
                          nbatch=nbatch,
                          direction=direction,
                          inplace=inplace,
                          real=real,
                          precision=precision)

def md():
    """Molecular dynamics suite."""

    yield from default_length_params("md", lengths['md'], 10)

def qa():
    """AMD QA suite."""

    for length1 in [8192, 10752, 18816, 21504, 32256, 43008, 16384, 19683, 15625, 16807]:
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

    yield Problem((336,336,56),
                  tag=mktag('qa3', 3, 'double', -1, False, False),
                  nbatch=1,
                  direction=-1,
                  inplace=False,
                  real=False,
                  precision='double')

    for length3 in lengths['md']:
        for direction in [-1, 1]:
            yield Problem(length3,
                          tag=mktag('qa3md', 3, 'single', direction, False, True),
                          nbatch=1,
                          direction=direction,
                          inplace=False,
                          real=True,
                          precision='single')

    for length in lengths['qa1d10b']:
        yield Problem([length],
                      tag=mktag("qa1d10b", 1, 'single', -1, True, False),
                      nbatch=10,
                      direction=-1,
                      inplace=True,
                      real=False,
                      precision='single')

    for length2 in lengths['qa2d10b']:
        yield Problem(length2,
                      tag=mktag("qa2d10b", 2, 'single', -1, True, False),
                      nbatch=10,
                      direction=-1,
                      inplace=True,
                      real=False,
                      precision='single')

    for length3 in lengths['qa3d10b']:
        yield Problem(length3,
                      tag=mktag("qa3d10b", 3, 'single', -1, True, False),
                      nbatch=10,
                      direction=-1,
                      inplace=True,
                      real=False,
                      precision='single')

    for length3 in lengths['qaReal3d10b']:
        for direction in [-1, 1]:
            yield Problem(length3,
                          tag=mktag("qaReal3d10b", 3, 'single', direction, False, True),
                          nbatch=10,
                          direction=direction,
                          inplace=False,
                          real=True,
                          precision='single')

def misc2d():
    """Miscellaneous 2D sizes."""

    yield from default_length_params("misc2d", lengths['misc2d'], 1)

def misc3d():
    """Miscellaneous 3D sizes."""

    yield from default_length_params("misc3d", lengths['misc3d'], 1)

def simpleL1D():
    """Basic C2C Large 1D sizes."""

    yield from default_length_params("C2C_L1D", lengths['simpleL1D'], 8000, reals=[False])

def large1d():
    """Large 1D sizes."""

    yield from default_length_params("large1d", lengths['large1d'], 10000)


def generated1d():
    """Explicitly generated 1D lengths."""

    yield from default_length_params("generated1d", lengths['generated'], 1000)

def generated2d():
    """Explicitly generated 2D lengths."""

    lengths2d  = list(filter(lambda x: x <= 1024, lengths['generated']))
    yield from default_length_params("generated2d", lengths2d, 100)

def generated3d():
    """Explicitly generated 3D lengths."""

    lengths3d  = list(filter(lambda x: x <= 512, lengths['generated']))
    yield from default_length_params("generated3d", lengths3d, 1)

def prime():
    """Large selection of prime lengths."""

    yield from default_length_params("prime", list(sympy.sieve.primerange(11, 1000)), 10000)


def mixed1d():
    """Mixed 1D lengths."""

    yield from default_length_params("mixed", lengths['mixed'], 10000)

def prime_limited():
    """Limited selection of prime lengths for regular testing."""

    yield from default_length_params("prime", [23, 521, 997], 10000)

def benchmarks():
    """Benchmarks: XXX"""

    pow2 = [2**k for k in range(30)]
    pow3 = [3**k for k in range(19)]
    minmax = {
        1: (256, 536870912),
        2: (64, 32768),
        3: (16, 1024),
    }

    all_lengths    = sorted(pow2 + pow3)
    dimensions = [1, 2, 3]

    for dimension in dimensions:
        min1, max1 = minmax[dimension]
        lengths = [(3*[length])[:dimension] for length in all_lengths if min1 <= length <= max1]
        yield from default_length_params('benchmark', lengths, 1)

def all():
    """All suites run during regular testing."""

    yield from benchmarks()

    # pow 5, 7 (benchmarks does pow 2, 3)
    yield from default_length_params("pow5", [ 5**k for k in range(4,8) ], 5000)
    yield from default_length_params("pow7", [ 7**k for k in range(3,7) ], 5000)

    yield from md()

    yield from prime_limited()

    yield from large1d()
    yield from misc2d()
    yield from misc3d()
