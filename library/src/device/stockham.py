"""Stockham kernel generator."""

import functools
import sys

from collections import namedtuple
from math import ceil
from pathlib import Path
from types import SimpleNamespace as NS
from enum import Enum

from generator import *


class Tiling(Enum):
    linear = 0
    cc = 1

str2enum = {'CS_KERNEL_STOCKHAM':           Tiling.linear,
            'CS_KERNEL_STOCKHAM_BLOCK_CC':  Tiling.cc
           }

def tiling_str_to_enum(scheme):
    return str2enum[scheme]
def tiling_enum_to_str(tiling):
    return [_scheme for _scheme, _tiling in str2enum.items() if _tiling == tiling][0]

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


def quantize(n, granularity):
    """Round up a number 'n' to the next integer multiple of 'granularity'"""
    return granularity * ((n-1)//granularity + 1)


def get_launch_params(factors, flavour='uwide', bytes_per_element=16, lds_byte_limit=32 * 1024, threads_per_block=256, **kwargs):
    """Return kernel launch parameters.

    Computes the maximum number of batches-per-block without:
    - going over 'lds_byte_limit' (32KiB by default) per block
    - going beyond 'threads_per_block' threads per block.
    """
    thread_granularity = 1

    length = product(factors)
    bytes_per_batch = length * bytes_per_element

    if flavour == 'uwide':
        threads_per_transform = length // min(factors)
    elif flavour == 'wide':
        threads_per_transform = length // max(factors)

    bpb = lds_byte_limit // bytes_per_batch
    while threads_per_transform * bpb > threads_per_block:
        bpb -= 1
    return LaunchParams(bpb, quantize(threads_per_transform * bpb, thread_granularity), threads_per_transform)


def get_callback_args():
    cb_args = []
    cb_args.append(Variable('load_cb_fn', 'void', array=True, restrict=True))
    cb_args.append(Variable('load_cb_data', 'void', array=True, restrict=True))
    cb_args.append(Variable('load_cb_lds_bytes', 'uint32_t'))
    cb_args.append(Variable('store_cb_fn', 'void', array=True, restrict=True))
    cb_args.append(Variable('store_cb_data', 'void', array=True, restrict=True))
    return ArgumentList(*cb_args)


def stockham_uwide_device(factors, **kwargs):
    """Stockham kernel.

    Each thread does at-most one butterfly.
    """
    length = product(factors)
    tiling = kwargs.get('tiling')
    params = get_launch_params(factors, **kwargs)

    # template params
    sb          = Variable('sb', 'StrideBin')
    cbtype      = Variable('cbtype', 'CallbackType')
    Ttwd_large  = Variable('Ttwd_large', 'bool')
    scalar_type = Variable('scalar_type', 'typename')
    ltwd_base   = Variable('LTBase', 'size_t')

    # arguments
    Z           = Variable('buf', 'scalar_type', array=True)
    X           = Variable('lds', 'scalar_type', array=True)
    T           = Variable('twiddles', 'const scalar_type', array=True)
    twd_large   = Variable('twd_large', 'const scalar_type', array=True)
    stride      = Variable('stride', 'size_t')
    offset      = Variable('offset', 'size_t')
    cb_args     = get_callback_args()
    offset_lds  = Variable('offset_lds', 'unsigned int') # lds address space is limited to 64K
    trans_local = Variable('trans_local', 'size_t')

    # locals
    thread    = Variable('thread', 'int')
    thread_id = Variable('threadIdx.x', 'int')
    W         = Variable('W', 'scalar_type')
    t         = Variable('t', 'scalar_type')
    R         = Variable('R', 'scalar_type', max(factors))

    body = StatementList()
    body += Declarations(thread, R, W, t)
    body += CallbackDeclaration()

    body += LineBreak()
    body += Assign(thread, thread_id % (length // min(factors)))
    body += LineBreak()

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

        if npass == len(factors) - 1 and tiling == Tiling.cc:
            body += CommentLines('large twiddle multiplication')
            stmts = StatementList()
            for w in range(width):
                idx = B(B(thread % cumheight) + w * cumheight) * trans_local
                stmts += Assign(W, InlineCall('TW2step',
                         arguments=ArgumentList(twd_large, idx),
                         templates = TemplateList(scalar_type, ltwd_base)
                         ))
                stmts += Assign(t.x, W.x * R[w].x - W.y * R[w].y)
                stmts += Assign(t.y, W.y * R[w].x + W.x * R[w].y)
                stmts += Assign(R[w], t)
            body += If(Ttwd_large, stmts)

        body += CommentLines('store lds')
        body += SyncThreads()
        stmts = StatementList()
        for w in range(width):
            idx = offset_lds + B(thread / cumheight) * (width * cumheight) + thread % cumheight + w * cumheight
            stmts += Assign(X[idx], R[w])
        body += If(thread < length // width, stmts)
        body += LineBreak()

    body += LineBreak()

    template_list=TemplateList(scalar_type, sb, cbtype)
    argument_list=ArgumentList(Z, X, T, stride, offset, offset_lds) + cb_args
    if tiling == Tiling.cc:
        template_list += TemplateList(Ttwd_large, ltwd_base)
        argument_list += ArgumentList(twd_large, trans_local)
    return Function(f'forward_length{length}_{tiling.name}_device',
                    arguments=argument_list,
                    templates=template_list,
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
    cbtype      = Variable('cbtype', 'CallbackType')
    Z           = Variable('buf', 'scalar_type', array=True)
    X           = Variable('lds', 'scalar_type', array=True)
    T           = Variable('twiddles', 'const scalar_type', array=True)
    stride      = Variable('stride', 'size_t')
    offset      = Variable('offset', 'size_t')
    offset_lds  = Variable('offset_lds', 'size_t')
    cb_args     = get_callback_args()

    # locals
    thread    = Variable('thread', 'int')
    thread_id = Variable('threadIdx.x', 'int')
    W         = Variable('W', 'scalar_type')
    t         = Variable('t', 'scalar_type')
    R         = Variable('R', 'scalar_type', 2*max(factors))
    stride_0  = Variable('stride_0', 'const size_t')

    height0 = length // max(factors)

    def load_global():
        stmts = StatementList()
        stmts += Assign(thread, thread_id % height0)
        stmts += LineBreak()
        for b in range(max(factors)):
            idx = thread + b * height0
            stmts += Assign(X[offset_lds + idx], LoadGlobal(Z, offset + idx))
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
            stmts += StoreGlobal(Z, offset + idx, X[offset_lds + idx])
        return stmts


    def add_work(codelet):
        if nsubpasses == 1 or nsubpass < nsubpasses - 1:
            return codelet()
        needs_work = thread_id % height0 + nsubpass * height0 < length // width
        return If(needs_work, codelet())


    body = StatementList()
    body += Declarations(thread, R, W, t)
    body += CallbackDeclaration()
    #body += Declaration(stride_0, Ternary(unit_stride, 1, stride))
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
                    arguments=ArgumentList(Z, X, T, stride, offset, offset_lds) + cb_args,
                    templates=TemplateList(scalar_type, sb, cbtype),
                    body=body,
                    qualifier='__device__')


def stockham_global(factors, **kwargs):
    """Global Stockham function."""
    length     = product(factors)
    tiling     = kwargs.get('tiling')
    use_3steps = kwargs.get('3steps')
    params     = get_launch_params(factors, **kwargs)

    # template params
    scalar_type = Variable('scalar_type', 'typename')
    cbtype      = Variable('cbtype', 'CallbackType')
    sb          = Variable('sb', 'StrideBin')
    Ttwd_large  = Variable('Ttwd_large', 'bool')
    ltwd_base   = Variable('LTBase', 'size_t', value=8)

    # arguments
    buf         = Variable('buf', 'scalar_type', array=True, restrict=True)
    twiddles    = Variable('twiddles', 'const scalar_type', array=True, restrict=True)
    cb_args     = get_callback_args()
    twd_large   = Variable('twd_large', 'const scalar_type', array=True, restrict=True)
    dim         = Variable('dim', 'const size_t')
    lengths     = Variable('lengths', 'const size_t', array=True, restrict=True)
    stride      = Variable('stride', 'const size_t', array=True, restrict=True)
    nbatch      = Variable('nbatch', 'const size_t')

    # locals
    lds        = Variable('lds', '__shared__ scalar_type', size=length * params.transforms_per_block)
    block_id   = Variable('blockIdx.x')
    thread_id  = Variable('threadIdx.x')
    offset     = Variable('offset', 'size_t', value=0)
    offset_lds = Variable('offset_lds', 'unsigned int') # lds address space is limited to 64K
    batch      = Variable('batch', 'size_t')
    transform  = Variable('transform', 'size_t')
    remaining  = Variable('remaining', 'size_t')
    plength    = Variable('plength', 'size_t', value=1)
    d          = Variable('d', 'size_t')
    i_d        = Variable('i_d', 'size_t')
    stride_0   = Variable('stride_0', 'const size_t')

    unit_stride = sb == 'SB_UNIT'

    body = StatementList()
    body += CommentLines(
        f'this kernel:',
        f'  uses {params.threads_per_transform} threads per transform',
        f'  does {params.transforms_per_block} transforms per thread block',
        f'therefore it should be called with {params.threads_per_block} threads per thread block')
    body += Declarations(lds, offset, offset_lds, batch, transform, remaining, plength, d, i_d)
    body += CallbackDeclaration()

    body += LineBreak()
    body += Declaration(stride_0.name, stride_0.type, value=Ternary(unit_stride, 1, stride[0]))
    if tiling == Tiling.linear:
        body += Assign(transform, block_id * params.transforms_per_block + thread_id / (length // min(factors)))
        body += Assign(remaining, transform)
        body += For(InlineAssign(d, 1), d < dim, Increment(d),
                    StatementList(
                        Assign(plength, plength * lengths[d]),
                        Assign(i_d, remaining % lengths[d]),
                        Assign(remaining, remaining / lengths[d]),
                        Assign(offset, offset + i_d * stride[d])))
        body += LineBreak()
        body += Assign(batch, transform / plength)
    elif tiling == Tiling.cc:

        ltwd_entries  = Multiply(B(ShiftLeft(1, ltwd_base)), 3) # (1 << LTBase) * 3
        ltwdLDS_cond  = And(Ttwd_large, Less(ltwd_base, 8)) # Ttwd_large && LTBase < 8
        large_twd_lds = Variable('large_twd_lds', '__shared__ scalar_type', size=Ternary(ltwdLDS_cond, ltwd_entries, 0))
        ltwd_id       = Variable('ltwd_id', 'size_t', value=thread_id)

        body += LineBreak()
        body += Declarations(large_twd_lds)
        body += If(ltwdLDS_cond,
                    StatementList(
                        Declaration(ltwd_id.name, ltwd_id.type, value=ltwd_id.value),
                        While(Less(ltwd_id, ltwd_entries),
                            StatementList(
                                Assign(large_twd_lds[ltwd_id], twd_large[ltwd_id]),
                                AddAssign(ltwd_id, params.threads_per_block)
                            )
                        )
                    )
                )

        i_1      = Variable('i_1', 'size_t')
        length_1 = Variable('length_1', 'size_t')
        body += LineBreak()
        body += CommentLines('calculate offset for each tile:',
                             '  i_1      now means index of the tile along dim1',
                             '  length_1 now means number of tiles along dim1')
        body += Declarations(i_1, length_1)
        body += Assign(length_1, B(lengths[1] - 1) / params.transforms_per_block + 1)
        body += Assign(plength, length_1)
        body += Assign(i_1, block_id % length_1)
        body += Assign(remaining, block_id / length_1)
        body += Assign(offset, i_1 * params.transforms_per_block * stride[1])
        body += For(InlineAssign(d, 2), d < dim, Increment(d),
                    StatementList(
                        Assign(plength, plength * lengths[d]),
                        Assign(i_d, remaining % lengths[d]),
                        Assign(remaining, remaining / lengths[d]),
                        Assign(offset, offset + i_d * stride[d])))
        body += LineBreak()
        body += Assign(transform, i_1 * params.transforms_per_block + thread_id / (length // min(factors)))
        body += Assign(batch, block_id / plength)

    body += If(GreaterEqual(batch, nbatch), [ReturnStatement()])
    body += LineBreak()
    body += Assign(offset, offset + batch * stride[dim])
    body += Assign(offset_lds, length * B(transform % params.transforms_per_block))
    body += LineBreak()

    body += CommentLines('load global')
    if tiling == Tiling.linear:
        width = min(factors)
        thread = Variable('thread', 'int')
        body += Declarations(thread)
        body += Assign(thread, thread_id % (length // min(factors)))
        for w in range(width):
            idx = thread + w * (length // width)
            body += Assign(lds[offset_lds + idx], LoadGlobal(buf, offset + B(idx) * stride_0))
    elif tiling == Tiling.cc:
        edge = Variable('edge', 'bool')
        tid1 = Variable('tid1', 'size_t')
        tid0 = Variable('tid0', 'size_t')
        body += Declarations(edge, tid0, tid1)
        stripmine_w = params.transforms_per_block
        stripmine_h = params.threads_per_block // stripmine_w
        body += ConditionalAssign(edge,
                                  Greater(B(i_1+1)*params.transforms_per_block, lengths[1]),
                                  'true', 'false')
        body += Assign(tid1, thread_id % stripmine_w) # tid0 walks the columns; tid1 walks the rows
        body += Assign(tid0, thread_id / stripmine_w)
        stride_lds = length + kwargs.get('lds_padding', 0)
        offset_tile_rbuf = lambda i : tid1 * stride[1]  + B(tid0 + i * stripmine_h) * stride_0
        offset_tile_wlds = lambda i : tid1 * stride_lds + B(tid0 + i * stripmine_h) * 1
        pred, stmts = StatementList(), StatementList()
        pred = i_1 * params.transforms_per_block + tid1 < lengths[1]
        for i in range(length//stripmine_h):
            stmts += Assign(lds[offset_tile_wlds(i)], LoadGlobal(buf, offset + offset_tile_rbuf(i)))
        body += If(Not(edge), stmts)
        body += If(edge, If(pred, stmts))

    body += LineBreak()

    if tiling == Tiling.cc:
        twd_large_arguments = Ternary(ltwdLDS_cond, large_twd_lds, twd_large)

    body += Call(f'forward_length{length}_{tiling.name}_device',
                 arguments=ArgumentList(buf, lds, twiddles, stride_0, offset, offset_lds) + cb_args + ([twd_large_arguments, transform] if tiling==Tiling.cc else []),
                 templates=TemplateList(scalar_type, sb, cbtype) + ([Ttwd_large, ltwd_base] if tiling==Tiling.cc else []))
    body += LineBreak()

    body += CommentLines('store global')
    body += SyncThreads()
    if tiling == Tiling.linear:
        width = min(factors)
        stmts = StatementList()
        for w in range(width):
            idx = thread + w * (length // width)
            stmts += StoreGlobal(buf, offset + B(idx) * stride_0, lds[offset_lds + idx])
        body += If(thread < length // width, stmts)
    elif tiling == Tiling.cc:
        offset_tile_wbuf = offset_tile_rbuf
        offset_tile_rlds = offset_tile_wlds
        pred, stmts = StatementList(), StatementList()
        pred = i_1 * params.transforms_per_block + tid1 < lengths[1]
        for i in range(length//stripmine_h):
            stmts += StoreGlobal(buf, offset + offset_tile_wbuf(i), lds[offset_tile_rlds(i)])
        body += If(Not(edge), stmts)
        body += If(edge, If(pred, stmts))

    template_list = TemplateList(scalar_type, sb, cbtype)
    argument_list = ArgumentList(twiddles, dim, lengths, stride, nbatch) + cb_args + ArgumentList(buf)
    if tiling == Tiling.cc:
        template_list += TemplateList(Ttwd_large, ltwd_base)
        argument_list = ArgumentList(twiddles, twd_large, dim, lengths, stride, nbatch) + cb_args + ArgumentList(buf)
    return Function(name=f'forward_length{length}_{tiling.name}',
                    qualifier=f'__global__ __launch_bounds__({params.threads_per_block})',
                    templates=template_list,
                    arguments=argument_list,
                    meta=NS(factors=factors,
                            length=product(factors),
                            transforms_per_block=params.transforms_per_block,
                            threads_per_block=params.threads_per_block,
                            scheme=tiling_enum_to_str(tiling),
                            use_3steps_large_twd=use_3steps,
                            pool=None),
                    body=body)

# XXX: move this to generator
def make_variants(kdevice, kglobal):
    """Given in-place complex-interleaved kernels, create all other variations.

    The ASTs in 'kglobal' and 'kdevice' are assumed to be in-place,
    complex-interleaved kernels.

    Return out-of-place and planar variations.
    """
    op_names = ['buf', 'stride', 'stride_0', 'offset']

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

    kdevice = make_inverse(kdevice, ['twiddles', 'TW2step'])
    kglobal = make_inverse(kglobal, ['twiddles', 'TW2step'])

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
    cbtype      = Variable('cbtype', 'CallbackType')
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
                 templates = TemplateList(scalar_type, sb, cbtype),
                 launch_params = ArgumentList(nblocks, params.threads_per_block))

    return Function(name = f'forward_length{length}_launch',
                    templates = TemplateList(scalar_type, cbtype),
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

    kwargs['factors'] = factors = kwargs.get('factors', stockham_default_factors(length))
    kwargs.pop('factors')

    # assert that factors multiply out to the length
    if functools.reduce(lambda x, y: x*y, factors) != length:
        raise RuntimeError("invalid factors {} for length {}".format(factors, length))

    defualt_3steps = {
        'sp':'false',
        'dp':'false'
    }
    kwargs['3steps'] = kwargs.get('use_3steps_large_twd', defualt_3steps)

    kwargs['tiling'] = tiling_str_to_enum(kwargs.get('scheme'))
    flavour = kwargs.get('flavour', 'uwide')
    if flavour == 'uwide':
        kdevice = stockham_uwide_device(factors, **kwargs)
    if flavour == 'wide':
        kdevice = stockham_wide_device(factors, **kwargs)
    kglobal = stockham_global(factors, **kwargs)
    return (kdevice, kglobal)
