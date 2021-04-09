"""Stockham kernel generator."""

import functools
import sys

from collections import namedtuple
from math import ceil
from pathlib import Path
from types import SimpleNamespace as NS

from generator import *


#
# Stockham FFTs!
#

LaunchParams = namedtuple('LaunchParams', ['transforms_per_block', 'threads_per_block', 'threads_per_transform'])


def kernel_launch_name(length, precision):
    """Return kernel name."""
    return f'rocfft_internal_dfn_{precision}_ci_ci_stoc_{length}'


def product(factors):
    """Return the product of the factors."""
    if factors:
        return functools.reduce(lambda a, b: a * b, factors)
    return 1


def get_launch_params(factors, flavour='uwide', bytes_per_element=16, lds_byte_limit=32 * 1024, threads_per_block=256):
    """Return kernel launch parameters.

    Computes the maximum number of batches-per-block without:
    - going over 'lds_byte_limit' (32KiB by default) per block
    - going beyond 'threads_per_block' threads per block.
    """
    length = product(factors)
    bytes_per_batch = length * bytes_per_element

    if flavour == 'uwide':
        threads_per_transform = length // min(factors)
    elif flavour == 'wide':
        threads_per_transform = length // max(factors)

    bpb = lds_byte_limit // bytes_per_batch
    while threads_per_transform * bpb > threads_per_block:
        bpb -= 1
    return LaunchParams(bpb, threads_per_transform * bpb, threads_per_transform)


def stockham_uwide_device(factors, **kwargs):
    """Stockham kernel.

    Each thread does at-most one butterfly.
    """
    length = product(factors)

    # arguments
    scalar_type = Variable('scalar_type', 'typename')
    sb          = Variable('sb', 'StrideBin')
    Z           = Variable('buf', 'scalar_type', array=True)
    X           = Variable('lds', 'scalar_type', array=True)
    T           = Variable('twiddles', 'const scalar_type', array=True)
    stride      = Variable('stride', 'size_t')
    offset      = Variable('offset', 'size_t')
    offset_lds  = Variable('offset_lds', 'size_t')

    # locals
    thread    = Variable('thread', 'int')
    thread_id = Variable('threadIdx.x', 'int')
    W         = Variable('W', 'scalar_type')
    t         = Variable('t', 'scalar_type')
    R         = Variable('R', 'scalar_type', max(factors))
    lstride   = Variable('lstride', 'size_t const')

    unit_stride = sb == 'SB_UNIT'

    body = StatementList()
    body += Declarations(thread, R, W, t)
    body += Declaration(lstride.name, lstride.type, None, Ternary(unit_stride, 1, stride))

    body += LineBreak()
    body += Assign(thread, thread_id % (length // min(factors)))
    body += LineBreak()

    body += CommentLines('load global')
    width = min(factors)
    for b in range(width):
        idx = thread + b * (length // width)
        body += Assign(X[offset_lds + idx], Z[offset + B(idx) * lstride])

    #
    # transform
    #
    for npass, width in enumerate(factors):
        cumheight = product(factors[:npass])

        body += LineBreak()
        body += CommentLines(f'pass {npass}')

        body += LineBreak()
        body += SyncThreads()
        body += LineBreak()

        body += CommentLines('load lds')
        for w in range(width):
            idx = offset_lds + thread + length // width * w
            body += Assign(R[w], X[idx])
        body += LineBreak()

        if npass > 0:
            body += CommentLines('twiddle')
            for w in range(1, width):
                tidx = cumheight - 1 + w - 1 + (width - 1) * B(thread % cumheight)
                body += Assign(W, T[tidx])
                body += Assign(t.x, W.x * R[w].x - W.y * R[w].y)
                body += Assign(t.y, W.y * R[w].x + W.x * R[w].y)
                body += Assign(R[w], t)
            body += LineBreak()

        body += CommentLines('butterfly')
        body += Call(name=f'FwdRad{width}B1',
                     arguments=ArgumentList(*[R[w].address() for w in range(width)]))
        body += LineBreak()

        if npass == len(factors) - 1:

            body += CommentLines('store global')
            store = StatementList()
            width = factors[-1]
            for w in range(width):
                idx = thread + w * (length // width)
                store += Assign(Z[offset + B(idx) * lstride], R[w])
            body += If(thread < length // width, store)

        else:

            body += CommentLines('store lds')
            body += SyncThreads()
            stmts = StatementList()
            for w in range(width):
                idx = offset_lds + B(thread / cumheight) * (width * cumheight) + thread % cumheight + w * cumheight
                stmts += Assign(X[idx], R[w])
            body += If(thread < length // width, stmts)
            body += LineBreak()

    body += LineBreak()

    return Function(f'forward_length{length}_device',
                    arguments=ArgumentList(Z, X, T, stride, offset, offset_lds),
                    templates=TemplateList(scalar_type, sb),
                    body=body,
                    qualifier='__device__')


def stockham_wide_device(factors, **kwargs):
    """Stockham kernel.

    Each thread does at-least one butterfly.
    """
    length = product(factors)

    # arguments
    scalar_type = Variable('scalar_type', 'typename')
    sb          = Variable('sb', 'StrideBin')
    Z           = Variable('buf', 'scalar_type', array=True)
    X           = Variable('lds', 'scalar_type', array=True)
    T           = Variable('twiddles', 'const scalar_type', array=True)
    stride      = Variable('stride', 'size_t')
    offset      = Variable('offset', 'size_t')
    offset_lds  = Variable('offset_lds', 'size_t')

    # locals
    thread    = Variable('thread', 'int')
    thread_id = Variable('threadIdx.x', 'int')
    W         = Variable('W', 'scalar_type')
    t         = Variable('t', 'scalar_type')
    R         = Variable('R', 'scalar_type', 2*max(factors))
    lstride   = Variable('lstride', 'size_t const')

    height0 = length // max(factors)

    def load_global():
        stmts = StatementList()
        stmts += Assign(thread, thread_id % height0)
        stmts += LineBreak()
        for b in range(max(factors)):
            idx = thread + b * height0
            stmts += Assign(X[offset_lds + idx], Z[offset + idx])
        return stmts

    def load_lds():
        stmts = StatementList()
        stmts += Assign(thread, thread_id % height0 + nsubpass * height0)
        for w in range(width):
            idx = offset_lds + thread + length//width * w
            stmts += Assign(R[nsubpass*width+w], X[idx])
        return stmts

    def twiddle():
        stmts = StatementList()
        stmts += Assign(thread, thread_id % height0 + nsubpass * height0)
        for w in range(1, width):
            tidx = cumheight - 1 + w - 1 + (width - 1) * B(thread % cumheight)
            ridx = nsubpass*width + w
            stmts += Assign(W, T[tidx])
            stmts += Assign(t.x, W.x * R[ridx].x - W.y * R[ridx].y)
            stmts += Assign(t.y, W.y * R[ridx].x + W.x * R[ridx].y)
            stmts += Assign(R[ridx], t)
        return stmts

    def butterfly():
        stmts = StatementList()
        stmts += Call(name=f'FwdRad{width}B1',
                    arguments=ArgumentList(*[R[nsubpass*width+w].address() for w in range(width)]))
        return stmts

    def store_lds():
        stmts = StatementList()
        stmts += Assign(thread, thread_id % height0 + nsubpass * height0)
        stmts += LineBreak()
        for w in range(width):
            idx = offset_lds + B(thread / cumheight) * (width * cumheight) + thread % cumheight + w * cumheight
            stmts += Assign(X[idx], R[nsubpass*width+w])
        stmts += LineBreak()
        return stmts


    def store_global():
        stmts = StatementList()
        stmts += SyncThreads()
        stmts += Assign(thread, thread_id % height0)
        for b in range(max(factors)):
            idx = thread + b * height0
            stmts += Assign(Z[offset + idx], X[offset_lds + idx])
        return stmts


    def add_work(codelet):
        if nsubpasses == 1 or nsubpass < nsubpasses - 1:
            return codelet()
        needs_work = thread_id % height0 + nsubpass * height0 < length // width
        return If(needs_work, codelet())


    body = StatementList()
    body += Declarations(thread, R, W, t)
    #body += Declaration(lstride, Ternary(unit_stride, 1, stride))
    body += LineBreak()
    body += CommentLines('load global')
    body += load_global()
    body += LineBreak()
    body += SyncThreads()
    body += LineBreak()

    #
    # transform
    #
    for npass, width in enumerate(factors):
        cumheight = product(factors[:npass])
        nsubpasses = ceil(max(factors) / factors[npass])

        body += CommentLines(f'pass {npass}')

        if npass > 0:
            body += SyncThreads()
            body += LineBreak()


        body += CommentLines('load lds')
        for nsubpass in range(nsubpasses):
            body += add_work(load_lds)
        body += LineBreak()
        if npass > 0:
            body += CommentLines('twiddle')
            for nsubpass in range(nsubpasses):
                body += add_work(twiddle)
            body += LineBreak()
        body += CommentLines('butterfly')
        for nsubpass in range(nsubpasses):
            body += add_work(butterfly)
        body += LineBreak()
        body += CommentLines('store lds')
        for nsubpass in range(nsubpasses):
            body += add_work(store_lds)

        body += LineBreak()

    body += CommentLines('store global')
    body += store_global()

    return Function(f'forward_length{length}_device',
                    arguments=ArgumentList(Z, X, T, stride, offset, offset_lds),
                    templates=TemplateList(scalar_type, sb),
                    body=body,
                    qualifier='__device__')


def stockham_global(factors, **kwargs):
    """Global Stockham function."""
    length = product(factors)
    params = get_launch_params(factors, **kwargs)

    # arguments
    scalar_type = Variable('scalar_type', 'typename')
    sb          = Variable('sb', 'StrideBin')
    buf         = Variable('buf', 'scalar_type', array=True, restrict=True)
    twiddles    = Variable('twiddles', 'const scalar_type', array=True, restrict=True)
    dim         = Variable('dim', 'const size_t')
    lengths     = Variable('lengths', 'const size_t', array=True)
    stride      = Variable('stride', 'const size_t', array=True)
    nbatch      = Variable('nbatch', 'const size_t')

    # locals
    lds        = Variable('lds', '__shared__ scalar_type', size=length * params.transforms_per_block)
    block_id   = Variable('blockIdx.x')
    thread_id  = Variable('threadIdx.x')
    offset     = Variable('offset', 'size_t')
    offset_lds = Variable('offset_lds', 'size_t')
    batch      = Variable('batch', 'size_t')
    transform  = Variable('transform', 'size_t')
    remaining  = Variable('remaining', 'size_t')
    plength    = Variable('plength', 'size_t')
    d          = Variable('d', 'size_t')
    i_d        = Variable('i_d', 'size_t')

    body = StatementList()
    body += CommentLines(
        f'this kernel:',
        f'  uses {params.threads_per_transform} threads per transform',
        f'  does {params.transforms_per_block} transforms per thread block',
        f'therefore it should be called with {params.threads_per_block} threads per thread block')
    body += Declarations(lds, offset, offset_lds, batch, transform, remaining, plength, d, i_d)

    body += LineBreak()
    body += Assign(transform, block_id * params.transforms_per_block + thread_id / params.threads_per_transform)
    body += LineBreak()
    body += Assign(offset, 0)
    body += Assign(plength, 1)
    body += Assign(remaining, transform)
    body += For(InlineAssign(d, 1), d < dim, Increment(d),
                StatementList(
                    Assign(plength, plength * lengths[d]),
                    Assign(i_d, remaining % lengths[d]),
                    Assign(remaining, remaining / lengths[d]),
                    Assign(offset, offset + i_d * stride[d])))
    body += LineBreak()
    body += Assign(batch, transform / plength)
    body += Assign(offset, offset +  batch * stride[dim])
    body += If(GreaterEqual(batch, nbatch), [ReturnStatement()])

    body += LineBreak()
    body += Assign(offset_lds, length * B(transform % params.transforms_per_block))
    body += Call(f'forward_length{length}_device',
                 arguments=ArgumentList(buf, lds, twiddles, stride[0], offset, offset_lds),
                 templates=TemplateList(scalar_type, sb))

    return Function(name=f'forward_length{length}',
                    qualifier='__global__',
                    templates=TemplateList(scalar_type, sb),
                    arguments=ArgumentList(twiddles, dim, lengths, stride, nbatch, buf),
                    meta=NS(factors=factors,
                            length=product(factors),
                            transforms_per_block=params.transforms_per_block,
                            threads_per_block=params.threads_per_block,
                            scheme='CS_KERNEL_STOCKHAM',
                            pool=None),
                    body=body)

# XXX: move this to generator
def make_variants(kdevice, kglobal):
    """Given in-place complex-interleaved kernels, create all other variations.

    The ASTs in 'kglobal' and 'kdevice' are assumed to be in-place,
    complex-interleaved kernels.

    Return out-of-place and planar variations.
    """
    op_names = ['buf', 'stride', 'lstride', 'offset']

    def rename(x, pre):
        if 'forward' in x or 'inverse' in x:
            return pre + x
        return x

    def rename_ip(x):
        return rename_functions(x, lambda n: rename(n, 'ip_'))

    def rename_op(x):
        return rename_functions(x, lambda n: rename(n, 'op_'))

    kernels = [
        # in-place, interleaved
        rename_ip(kdevice),
        rename_ip(kglobal),
        # in-place, planar
        rename_ip(make_planar(kdevice, 'buf')),
        rename_ip(make_planar(kglobal, 'buf')),
        # out-of-place, interleaved -> interleaved
        rename_op(make_out_of_place(kdevice, op_names)),
        rename_op(make_out_of_place(kglobal, op_names)),
        # out-of-place, interleaved -> planar
        rename_op(make_planar(make_out_of_place(kdevice, op_names), 'buf_out')),
        rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_out')),
        # out-of-place, planar -> interleaved
        rename_op(make_planar(make_out_of_place(kdevice, op_names), 'buf_in')),
        rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_in')),
        # out-of-place, planar -> planar
        rename_op(make_planar(make_planar(make_out_of_place(kdevice, op_names), 'buf_out'), 'buf_in')),
        rename_op(make_planar(make_planar(make_out_of_place(kglobal, op_names), 'buf_out'), 'buf_in')),
    ]

    kdevice = make_inverse(kdevice, ['twiddles'])
    kglobal = make_inverse(kglobal, ['twiddles'])

    kernels += [
        # in-place, interleaved
        rename_ip(kdevice),
        rename_ip(kglobal),
        # in-place, planar
        rename_ip(make_planar(kdevice, 'buf')),
        rename_ip(make_planar(kglobal, 'buf')),
        # out-of-place, interleaved -> interleaved
        rename_op(make_out_of_place(kdevice, op_names)),
        rename_op(make_out_of_place(kglobal, op_names)),
        # out-of-place, interleaved -> planar
        rename_op(make_planar(make_out_of_place(kdevice, op_names), 'buf_out')),
        rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_out')),
        # out-of-place, planar -> interleaved
        rename_op(make_planar(make_out_of_place(kdevice, op_names), 'buf_in')),
        rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_in')),
        # out-of-place, planar -> planar
        rename_op(make_planar(make_planar(make_out_of_place(kdevice, op_names), 'buf_out'), 'buf_in')),
        rename_op(make_planar(make_planar(make_out_of_place(kglobal, op_names), 'buf_out'), 'buf_in')),
    ]

    return kernels


def stockham_launch(factors, **kwargs):
    """Launch helper.  Not used by rocFFT proper."""

    length = product(factors)
    params = get_launch_params(factors, **kwargs)

    # arguments
    scalar_type = Variable('scalar_type', 'typename')
    sb          = Variable('SB_UNIT', 'StrideBin')
    inout       = Variable('inout', 'scalar_type', array=True)
    twiddles    = Variable('twiddles', 'const scalar_type', array=True)
    stride_in   = Variable('stride_in', 'size_t')
    stride_out  = Variable('stride_out', 'size_t')
    nbatch      = Variable('nbatch', 'size_t')
    kargs       = Variable('kargs', 'size_t*')

    # locals
    nblocks = Variable('nblocks', 'int')

    body = StatementList()
    body += Declarations(nblocks)
    body += Assign(nblocks, B(nbatch + (params.transforms_per_block - 1)) / params.transforms_per_block)
    body += Call(f'forward_length{length}',
                 arguments = ArgumentList(twiddles, 1, kargs, kargs+1, nbatch, inout),
                 templates = TemplateList(scalar_type, sb),
                 launch_params = ArgumentList(nblocks, params.threads_per_block))

    return Function(name = f'forward_length{length}_launch',
                    templates = TemplateList(scalar_type),
                    arguments = ArgumentList(inout, nbatch, twiddles, kargs, stride_in, stride_out),
                    body = body)

def stockham_default_factors(length):
    supported_radixes = [2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 16]
    supported_radixes.sort(reverse=True)

    remaining_length = length
    factors = []
    for f in supported_radixes:
        while remaining_length % f == 0:
            factors.append(f)
            remaining_length /= f

    if remaining_length != 1:
        raise RuntimeError("length {} not factorizable!".format(length))

    # default order of factors is ascending
    factors.sort()
    return factors

def stockham(length, **kwargs):
    """Generate Stockham kernels!

    Returns a list of (device, global) function pairs.
    """

    factors = kwargs.get('factors', stockham_default_factors(length))

    # assert that factors multiply out to the length
    if functools.reduce(lambda x, y: x*y, factors) != length:
        raise RuntimeError("invalid factors {} for length {}".format(factors, length))

    flavour = kwargs.get('flavour', 'uwide')
    if flavour == 'uwide':
        kdevice = stockham_uwide_device(factors, **kwargs)
    if flavour == 'wide':
        kdevice = stockham_wide_device(factors, **kwargs)
    kglobal = stockham_global(factors, **kwargs)
    return (kdevice, kglobal)
