"""Stockham kernel generator."""

import functools
import math
import sys

from collections import namedtuple
from math import ceil, floor
from pathlib import Path
from types import SimpleNamespace as NS
from enum import Enum

from generator import *


#
# Helpers
#

LaunchParams = namedtuple('LaunchParams', ['transforms_per_block',
                                           'threads_per_block',
                                           'threads_per_transform'])


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
    return granularity * ((n - 1) // granularity + 1)


def get_launch_params(factors,
                      flavour='uwide',
                      bytes_per_element=16,
                      lds_byte_limit=32 * 1024,
                      threads_per_block=256,
                      threads_per_transform=None,
                      **kwargs):
    """Return kernel launch parameters.

    Computes the maximum number of batches-per-block without:
    - going over 'lds_byte_limit' (32KiB by default) per block
    - going beyond 'threads_per_block' threads per block.
    """
    thread_granularity = 1

    length = product(factors)
    bytes_per_batch = length * bytes_per_element

    if threads_per_transform is None:
        threads_per_transform = 1
        for t in range(2, length):
            if t > threads_per_block:
                continue
            if length % t == 0:
                if all((length // t) % f == 0 for f in factors):
                    threads_per_transform = t

    batches_per_block = lds_byte_limit // bytes_per_batch
    while threads_per_transform * batches_per_block > threads_per_block:
        batches_per_block -= 1
    return LaunchParams(batches_per_block,
                        quantize(threads_per_transform * batches_per_block, thread_granularity),
                        threads_per_transform)


def get_callback_args():
    """Return callback argument list."""
    return ArgumentList(*[
        Variable('load_cb_fn', 'void', array=True, restrict=True),
        Variable('load_cb_data', 'void', array=True, restrict=True),
        Variable('load_cb_lds_bytes', 'uint32_t'),
        Variable('store_cb_fn', 'void', array=True, restrict=True),
        Variable('store_cb_data', 'void', array=True, restrict=True)])


def common_variables(length, params, nregisters):
    """Return namespace of common/frequent variables used in Stockham kernels."""
    kvars = NS(

        #
        # templates
        #

        scalar_type   = Variable('scalar_type', 'typename'),
        callback_type = Variable('cbtype', 'CallbackType'),
        stride_type   = Variable('sb', 'StrideBin'),
        embedded_type = Variable('ebtype', 'EmbeddedType'),

        #
        # arguments
        #

        # global input/ouput buffer
        buf = Variable('buf', 'scalar_type', array=True, restrict=True),

        # global twiddle table (stacked)
        twiddles = Variable('twiddles', 'const scalar_type', array=True, restrict=True),

        # rank/dimension of transform
        dim = Variable('dim', 'const size_t'),

        # transform lengths
        lengths = Variable('lengths', 'const size_t', array=True, restrict=True),

        # input/output array strides
        stride = Variable('stride', 'const size_t', array=True, restrict=True),

        # number of transforms/batches
        nbatch = Variable('nbatch', 'const size_t'),

        # the number of padding at the end of each row in lds
        lds_padding = Variable('lds_padding', 'const unsigned int'),

        # should the device function write to lds?
        write = Variable('write', 'bool'),

        #
        # locals
        #

        # lds storage buffer
        lds_uchar     = Variable('lds_uchar', 'unsigned char',
                            size='dynamic',  post_qualifier='__align__(sizeof(scalar_type))',
                            array=True, shared=True),
        lds = Variable('lds', 'scalar_type', array=True, restrict=True, pointer=True,
                       value = 'reinterpret_cast<scalar_type *>(lds_uchar)'), # FIXME: do it in AST properly),

        # hip thread block id
        block_id = Variable('blockIdx.x'),

        # hip thread id
        thread_id = Variable('threadIdx.x'),

        # thread within transform
        thread = Variable('thread', 'size_t'),

        # global input/output buffer offset to current transform
        offset = Variable('offset', 'size_t', value=0),

        # lds buffer offset to current transform
        offset_lds = Variable('offset_lds', 'unsigned int'),

        # current batch
        batch = Variable('batch', 'size_t'),

        # current transform
        transform = Variable('transform', 'size_t'),

        # stride between consecutive indexes
        stride0 = Variable('stride0', 'const size_t'),

        # stride between consecutive indexes in lds
        stride_lds = Variable('stride_lds', 'size_t'),

        # usually in device: const size_t lstride = (sb == SB_UNIT) ? 1 : stride_lds;
        # with this definition, the compiler knows that 'index * lstride' is trivial under SB_UNIT
        lstride = Variable('lstride', 'const size_t'),

        # twiddle value during twiddle application
        W = Variable('W', 'scalar_type'),

        # temporary register during twiddle application
        t = Variable('t', 'scalar_type'),

        # butterfly registers
        R = Variable('R', 'scalar_type', size=nregisters),
    )
    return kvars, kvars.__dict__


#
# Base helpers
#

class AdditionalArgumentMixin:
    """Add default template/argument list manipulation methods."""

    def add_templates(self, tlist, **kwargs):
        """Return list of extra template arguments."""
        return tlist

    def add_global_arguments(self, alist, **kwargs):
        """Return new function arguments to the global kernel function."""
        return alist

    def add_device_arguments(self, alist, **kwargs):
        """Return new function arguments to the device kernel function."""
        return alist

    def add_device_call_arguments(self, alist, **kwargs):
        """Return new function arguments for calling the device kernel function."""
        return alist


#
# Large twiddle table support
#

class StockhamLargeTwiddles(AdditionalArgumentMixin):
    """Base large twiddle table."""

    def load(self, length, params, **kwargs):
        return StatementList()

    def multiply(self, width, cumheight, **kwargs):
        return StatementList()


class StockhamLargeTwiddles2Step():
    """Two stage large twiddle table."""

    def __init__(self):
        self.apply_large_twiddle = Variable('apply_large_twiddle', 'bool')
        self.large_twiddle_base  = Variable('large_twiddle_base', 'size_t', value=8)
        self.large_twiddles      = Variable('large_twiddles', 'const scalar_type', array=True)
        self.trans_local         = Variable('trans_local', 'size_t')

    def add_templates(self, tlist, **kwargs):
        return tlist + TemplateList(self.apply_large_twiddle, self.large_twiddle_base)

    def add_global_arguments(self, alist, **kwargs):
        nargs = list(alist.args)
        nargs.insert(1, self.large_twiddles)
        return ArgumentList(*nargs)

    def add_device_arguments(self, alist, **kwargs):
        return alist + ArgumentList(self.large_twiddles, self.trans_local)

    def add_device_call_arguments(self, alist, transform=None, **kwargs):
        which = Ternary(And(self.apply_large_twiddle, self.large_twiddle_base < 8),
                        self.large_twd_lds, self.large_twiddles)
        return alist + ArgumentList(which, transform)

    def load(self, length, params,
             transform=None, dim=None,
             block_id=None, thread_id=None, lengths=None, stride=None, offset=None, batch=None,
             offset_lds=None,
             **kwargs):

        ltwd_id = Variable('ltwd_id', 'size_t', value=thread_id)
        ltwd_entries = Multiply(B(ShiftLeft(1, self.large_twiddle_base)), 3)
        ltwd_in_lds = And(self.apply_large_twiddle, Less(self.large_twiddle_base, 8))

        self.large_twd_lds = Variable('large_twd_lds', '__shared__ scalar_type',
                                      size=Ternary(ltwd_in_lds, ltwd_entries, 0))

        stmts = StatementList()
        stmts += Declarations(self.large_twd_lds)
        stmts += If(ltwd_in_lds,
                    StatementList(
                        Declaration(ltwd_id.name, ltwd_id.type, value=ltwd_id.value),
                        While(Less(ltwd_id, ltwd_entries),
                              StatementList(
                                  Assign(self.large_twd_lds[ltwd_id], self.large_twiddles[ltwd_id]),
                                  AddAssign(ltwd_id, params.threads_per_block)))))

        return stmts

    def multiply(self, width, cumheight,
                 W=None, t=None, R=None,
                 thread=None, scalar_type=None, **kwargs):
        stmts = StatementList()
        stmts += CommentLines('large twiddle multiplication')
        for w in range(width):
            # FIXME- using a .cast('type') would be graceful!
            #        Why casting to ((int)thread % 8) only when passing to TW2step ?
            #        This is completely based on the testing result. We observed that it can
            #        reduce a few vgprs (we don't know why since its behind the compiler)
            #        and avoid the drop of occuapancy (espcially for sbcc_len64_inverse)
            # idx = B(B(thread % cumheight) + w * cumheight) * self.trans_local
            idx = f'(((int){thread} % {cumheight}) + {w} * {cumheight}) * {self.trans_local}'
            stmts += Assign(W, InlineCall('TW2step',
                                          arguments=ArgumentList(self.large_twiddles, idx),
                                          templates=TemplateList(scalar_type, self.large_twiddle_base)))
            stmts += Assign(t, TwiddleMultiply(R[w], W))
            stmts += Assign(R[w], t)
        return If(self.apply_large_twiddle, stmts)


#
# Tilings
#

class StockhamTiling(AdditionalArgumentMixin):
    """Base tiling."""

    def __init__(self, factors, params, **kwargs):
        self.length = product(factors)
        self.factors = factors
        self.params = params

    def update_kernel_settings(self, kernel):
        pass

    def calculate_offsets(self, *args, **kwargs):
        """Return code to calculate batch and buffer offsets."""
        return StatementList()

    def load_from_global(self, *args, **kwargs):
        """Return code to load from global buffer to LDS."""
        return StatementList()

    def store_to_global(self, *args, **kwargs):
        """Return code to store LDS to global buffer."""
        return StatementList()

    def real2cmplx_pre_post(self, half_N, isPre, param, thread=None, thread_id=None, lds=None,
            offset_lds=None, twiddles=None, lds_padding=None, embedded_type=None, scalar_type=None,
            buf=None, offset=None, stride=None, **kwargs):
        """Return code to handle even-length real to complex pre/post-process in lds."""

        function_name = f'real_pre_process_kernel_inplace' if isPre else f'real_post_process_kernel_inplace'
        template_type = 'EmbeddedType::C2Real_PRE' if isPre else 'EmbeddedType::Real2C_POST'
        Ndiv4  = 'true' if half_N % 2 == 0 else 'false'
        quarter_N = half_N // 2
        if half_N % 2 == 1:
            quarter_N += 1

        stmts = StatementList()

        stmts += SyncThreads() # Todo: We might not have to sync here which depends on the access pattern
        stmts += LineBreak()

        # Todo: For case threads_per_transform == quarter_N, we could save one more "if" in the c2r/r2r kernels

        # if we have fewer threads per transform than quarter_N,
        # we need to call the pre/post function multiple times
        r2c_calls_per_transform = quarter_N // param.threads_per_transform
        if quarter_N % param.threads_per_transform > 0:
            r2c_calls_per_transform += 1
        for i in range(r2c_calls_per_transform):
             stmts += Call(function_name,
                    templates = TemplateList(scalar_type, Ndiv4),
                    arguments = ArgumentList(thread % quarter_N + i * param.threads_per_transform,
                        half_N - thread % quarter_N - i * param.threads_per_transform, quarter_N,
                        lds[offset_lds].address(),
                        0, twiddles[half_N].address()),)
        if (isPre):
            stmts += SyncThreads()
            stmts += LineBreak()

        return If(Equal(embedded_type, template_type), stmts)


class StockhamTilingRR(StockhamTiling):
    """Row/row tiling."""

    name = 'SBRR'

    def calculate_offsets(self, lengths=None, stride=None,
                          dim=None, transform=None, block_id=None, thread_id=None,
                          batch=None, offset=None, offset_lds=None, lds_padding=None, **kwargs):

        length, params = self.length, self.params

        d             = Variable('d', 'int')
        index_along_d = Variable('index_along_d', 'size_t')
        remaining     = Variable('remaining', 'size_t')

        stmts = StatementList()
        stmts += Declarations(remaining, index_along_d)
        stmts += Assign(transform, block_id * params.transforms_per_block + thread_id / params.threads_per_transform)
        stmts += Assign(remaining, transform)
        stmts += For(d.inline(1), d < dim, Increment(d),
                     StatementList(
                         Assign(index_along_d, remaining % lengths[d]),
                         Assign(remaining, remaining / lengths[d]),
                         Assign(offset, offset + index_along_d * stride[d])))
        stmts += Assign(batch, remaining)
        stmts += Assign(offset, offset + batch * stride[dim])
        stmts += Assign(offset_lds, B(length + lds_padding) * B(transform % params.transforms_per_block))

        return stmts

    def load_from_global(self, thread=None, thread_id=None, stride0=None,
                         buf=None, offset=None, lds=None, offset_lds=None,
                         embedded_type=None, **kwargs):
        length, params = self.length, self.params
        width  = params.threads_per_transform
        height = length // width
        stmts = StatementList()
        stmts += Assign(thread, thread_id % width)
        for w in range(height):
            idx = thread + w * width
            stmts += Assign(lds[offset_lds + idx], LoadGlobal(buf, offset + B(idx) * stride0))

        stmts += LineBreak()
        stmts += CommentLines('append extra global loading for C2Real pre-process only')
        stmts_c2real_pre = StatementList()
        stmts_c2real_pre += CommentLines('use the last thread of each transform to load one more element per row')
        stmts_c2real_pre += If(Equal(thread, params.threads_per_transform - 1),
            Assign(lds[offset_lds + thread + (height - 1) * width + 1],
            LoadGlobal(buf, offset + B(thread + (height - 1) * width + 1) * stride0)))
        stmts += If(Equal(embedded_type, 'EmbeddedType::C2Real_PRE'), stmts_c2real_pre)

        return stmts

    def store_to_global(self, thread=None, thread_id=None, stride0=None,
                        buf=None, offset=None, lds=None, offset_lds=None,
                        embedded_type=None, **kwargs):
        length, params = self.length, self.params
        width  = params.threads_per_transform
        height = length // width
        stmts = StatementList()
        for w in range(height):
            idx = thread + w * width
            stmts += StoreGlobal(buf, offset + B(idx) * stride0, lds[offset_lds + idx])

        stmts += LineBreak()
        stmts += CommentLines('append extra global write for Real2C post-process only')
        stmts_real2c_post = StatementList()
        stmts_real2c_post += CommentLines('use the last thread of each transform to write one more element per row')
        stmts_real2c_post += If(Equal(thread, params.threads_per_transform - 1),
            StoreGlobal(buf, offset + B(thread + (height - 1) * width + 1) * stride0,
            lds[offset_lds + thread + (height - 1) * width + 1]))
        stmts += If(Equal(embedded_type, 'EmbeddedType::Real2C_POST'), stmts_real2c_post)

        return stmts


class StockhamTilingCC(StockhamTiling):
    """Column/column tiling."""

    name = 'SBCC'

    def __init__(self, factors, params, **kwargs):
        super().__init__(factors, params, **kwargs)
        self.tile_index  = Variable('tile_index', 'size_t')
        self.tile_length = Variable('tile_length', 'size_t')
        self.edge        = Variable('edge', 'bool')
        self.tid1        = Variable('tid1', 'size_t')
        self.tid0        = Variable('tid0', 'size_t')

    def calculate_offsets(self, transform=None, dim=None,
                          block_id=None, thread_id=None, lengths=None, stride=None, offset=None, batch=None,
                          offset_lds=None,
                          **kwargs):

        length, params = self.length, self.params

        d             = Variable('d', 'int')
        index_along_d = Variable('index_along_d', 'size_t')
        remaining     = Variable('remaining', 'size_t')
        plength       = Variable('plength', 'size_t', value=1)

        stmts = StatementList()
        stmts += Declarations(self.tile_index, self.tile_length)
        stmts += LineBreak()
        stmts += CommentLines('calculate offset for each tile:',
                              '  tile_index  now means index of the tile along dim1',
                              '  tile_length now means number of tiles along dim1')
        stmts += Declarations(plength, remaining, index_along_d)
        stmts += Assign(self.tile_length, B(lengths[1] - 1) / params.transforms_per_block + 1)
        stmts += Assign(plength, self.tile_length)
        stmts += Assign(self.tile_index, block_id % self.tile_length)
        stmts += Assign(remaining, block_id / self.tile_length)
        stmts += Assign(offset, self.tile_index * params.transforms_per_block * stride[1])
        stmts += For(d.inline(2), d < dim, Increment(d),
                     StatementList(
                         Assign(plength, plength * lengths[d]),
                         Assign(index_along_d, remaining % lengths[d]),
                         Assign(remaining, remaining / lengths[d]),
                         Assign(offset, offset + index_along_d * stride[d])))
        stmts += LineBreak()
        stmts += Assign(transform, self.tile_index * params.transforms_per_block + thread_id / params.threads_per_transform)
        stmts += Assign(batch, block_id / plength)
        stmts += Assign(offset, offset + batch * stride[dim])
        stmts += Assign(offset_lds, length * B(transform % params.transforms_per_block))

        return stmts

    def load_from_global(self, buf=None, offset=None, lds=None,
                         lengths=None, thread_id=None, stride=None, stride0=None, **kwargs):

        length, params = self.length, self.params
        edge, tid0, tid1 = self.edge, self.tid0, self.tid1
        stripmine_w   = params.transforms_per_block
        stripmine_h   = params.threads_per_block // stripmine_w
        stride_lds    = length + kwargs.get('lds_padding', 0)  # XXX

        stmts = StatementList()
        stmts += Declarations(edge, tid0, tid1)
        stmts += ConditionalAssign(edge,
                                   Greater(B(self.tile_index + 1) * params.transforms_per_block, lengths[1]),
                                   'true', 'false')
        stmts += Assign(tid1, thread_id % stripmine_w)  # tid0 walks the columns; tid1 walks the rows
        stmts += Assign(tid0, thread_id / stripmine_w)
        offset_tile_rbuf = lambda i : tid1 * stride[1]  + B(tid0 + i * stripmine_h) * stride0
        offset_tile_wlds = lambda i : tid1 * stride_lds + B(tid0 + i * stripmine_h) * 1
        pred, tmp_stmts = StatementList(), StatementList()
        pred = self.tile_index * params.transforms_per_block + tid1 < lengths[1]
        for i in range(length // stripmine_h):
            tmp_stmts += Assign(lds[offset_tile_wlds(i)], LoadGlobal(buf, offset + offset_tile_rbuf(i)))

        stmts += If(Not(edge), tmp_stmts)
        stmts += If(edge, If(pred, tmp_stmts))

        return stmts

    def store_to_global(self, stride=None, stride0=None, lengths=None, buf=None, offset=None, lds=None,
                        **kwargs):

        length, params = self.length, self.params
        edge, tid0, tid1 = self.edge, self.tid0, self.tid1
        stripmine_w   = params.transforms_per_block
        stripmine_h   = params.threads_per_block // stripmine_w
        stride_lds    = length + kwargs.get('lds_padding', 0)  # XXX

        stmts = StatementList()
        offset_tile_rbuf = lambda i : tid1 * stride[1]  + B(tid0 + i * stripmine_h) * stride0
        offset_tile_wlds = lambda i : tid1 * stride_lds + B(tid0 + i * stripmine_h) * 1
        offset_tile_wbuf = offset_tile_rbuf
        offset_tile_rlds = offset_tile_wlds
        pred, tmp_stmts = StatementList(), StatementList()
        pred = self.tile_index * params.transforms_per_block + tid1 < lengths[1]
        for i in range(length // stripmine_h):
            tmp_stmts += StoreGlobal(buf, offset + offset_tile_wbuf(i), lds[offset_tile_rlds(i)])
        stmts += If(Not(edge), tmp_stmts)
        stmts += If(edge, If(pred, tmp_stmts))

        return stmts


class StockhamTilingRC(StockhamTiling):
    """Row/column tiling."""

    name = 'SBRC'

    def __init__(self, factors, params, **kwargs):
        super().__init__(factors, params, **kwargs)
        self.sbrc_type = Variable('sbrc_type', 'SBRC_TYPE')
        self.transpose_type = Variable('transpose_type', 'SBRC_TRANSPOSE_TYPE')
        self.lds_row_padding = Variable('lds_row_padding', 'unsigned int', value=0)
        self.rows_per_tile = 16
        self.n_device_calls = self.rows_per_tile // (self.params.threads_per_block // self.params.threads_per_transform)

    def update_kernel_settings(self, kernel):
        kernel.n_device_calls = self.n_device_calls
        kernel.lds_row_padding = self.lds_row_padding

    def add_templates(self, tlist, **kwargs):
        tlist = copy(tlist)
        tlist.args.insert(2, self.sbrc_type)
        tlist.args.insert(3, self.transpose_type)
        return tlist

    def add_global_arguments(self, alist, **kwargs):
        # only out-of-place, so replace strides...
        nargs = copy(alist)
        for i, arg in enumerate(alist.args):
            if arg.name == 'stride':
                si, so = copy(arg), copy(arg)
                si.name, so.name = 'stride_in', 'stride_out'
                nargs.args.pop(i)
                nargs.args.insert(i, so)
                nargs.args.insert(i, si)
                return nargs

    def calculate_offsets_2d(self,
                             tile=None, dim=None, transform=None, thread=None, lengths=None, offset_lds=None,
                             offset_in=None, offset_out=None, stride_in=None, stride_out=None,
                             **kwargs):

        length = self.length

        current_length = Variable('current_length', 'unsigned int')
        remaining = Variable('remaining', 'unsigned int')
        i, j = Variable('i', 'unsigned int'), Variable('j', 'unsigned int')

        offsets = StatementList()
        offsets += Declarations(current_length, remaining)
        offsets += Assign(remaining, tile)
        offsets += For(i.inline(dim), i > 2, Decrement(i),
                       StatementList(
                           Assign(current_length, 1),
                           For(j.inline(2), j < i, Increment(j),
                               StatementList(
                                   MultiplyAssign(current_length, lengths[j]))),
                           MultiplyAssign(current_length, lengths[1] / self.rows_per_tile),
                           AddAssign(offset_in, B(remaining / current_length) * stride_in[i]),
                           AddAssign(offset_out, B(remaining / current_length) * stride_out[i]),
                           Assign(remaining, remaining % current_length)))
        offsets += Assign(current_length, lengths[1] / self.rows_per_tile)
        offsets += AddAssign(offset_in, B(remaining / current_length) * stride_in[2])
        offsets += AddAssign(offset_in, B(remaining % current_length) * B(self.rows_per_tile * lengths[0]))

        offsets += AddAssign(offset_out, B(remaining / current_length) * stride_out[2])
        offsets += AddAssign(offset_out, B(remaining % current_length) * B(self.rows_per_tile * stride_out[1]))

        offsets += Assign(offset_lds, length * B(thread / self.rows_per_tile))

        return offsets

    def calculate_offsets_fft_trans_xy_z(self,
                                         tile=None, dim=None, transform=None, thread=None, lengths=None, offset_lds=None,
                                         offset_in=None, offset_out=None, stride_in=None, stride_out=None,
                                         **kwargs):

        length, transforms = self.length, self.params.transforms_per_block
        tiles_per_batch = B(lengths[1] * B(B(lengths[2] + self.rows_per_tile - 1) / self.rows_per_tile))

        def diagonal():
            tid = B(tile % lengths[1]) + length * B(B(tile % tiles_per_batch) / lengths[1])
            read_tile_y = B(tid % self.rows_per_tile)
            read_tile_x = B(B(B(tid / self.rows_per_tile) + read_tile_y) % length)
            write_tile_x = read_tile_y
            write_tile_y = read_tile_x
            return StatementList(
                AddAssign(offset_in, read_tile_x * stride_in[1]),
                AddAssign(offset_in, read_tile_y * self.rows_per_tile * stride_in[2]),
                AddAssign(offset_in, B(tile / tiles_per_batch) * stride_in[3]),
                AddAssign(offset_out, write_tile_x * self.rows_per_tile * stride_out[0]),
                AddAssign(offset_out, write_tile_y * stride_out[2]),
                AddAssign(offset_out, B(tile / tiles_per_batch) * stride_out[3]),
                Assign(offset_lds, B(thread / self.rows_per_tile) * length))

        def not_diagonal():
            read_tile_x = B(tile % lengths[1])
            read_tile_y = B(B(tile % tiles_per_batch) / lengths[1])
            write_tile_x = read_tile_y
            write_tile_y = read_tile_x
            return StatementList(
                AddAssign(offset_in, read_tile_x * stride_in[1]),
                AddAssign(offset_in, read_tile_y * self.rows_per_tile * stride_in[2]),
                AddAssign(offset_in, B(tile / tiles_per_batch) * stride_in[3]),
                AddAssign(offset_out, write_tile_x * self.rows_per_tile * stride_out[0]),
                AddAssign(offset_out, write_tile_y * stride_out[2]),
                AddAssign(offset_out, B(tile / tiles_per_batch) * stride_out[3]),
                Assign(offset_lds, B(thread / self.rows_per_tile) * length))

        return StatementList(
            IfElse(self.transpose_type == 'DIAGONAL', diagonal(), not_diagonal()))

    def calculate_offsets_fft_trans_z_xy(self,
                                         thread=None, tile=None, lengths=None, offset_lds=None,
                                         offset_in=None, offset_out=None, stride_in=None, stride_out=None,
                                         **kwargs):

        length = self.length

        tile_size_x = 1
        tile_size_y = B(lengths[1] * lengths[2] / self.rows_per_tile)
        tiles_per_batch = B(tile_size_x * tile_size_y)

        read_tile_x = 0
        read_tile_y = B(B(tile % tiles_per_batch) / tile_size_x)
        write_tile_x = read_tile_y
        write_tile_y = read_tile_x
        return StatementList(
            AddAssign(offset_in, read_tile_x * stride_in[1]),
            AddAssign(offset_in, read_tile_y * self.rows_per_tile * stride_in[1]),
            AddAssign(offset_in, B(tile / tiles_per_batch) * stride_in[3]),
            AddAssign(offset_out, write_tile_x * self.rows_per_tile * stride_out[0]),
            AddAssign(offset_out, write_tile_y * stride_out[3]),
            AddAssign(offset_out, B(tile / tiles_per_batch) * stride_out[3]),
            Assign(offset_lds, B(thread / self.rows_per_tile) * B(length + self.lds_row_padding)))


    def calculate_offsets(self,
                          transform=None, dim=None,
                          block_id=None, thread_id=None, lengths=None, stride=None, offset=None, batch=None,
                          offset_lds=None,
                          **kwargs):

        length, params = self.length, self.params

        tile = Variable('tile', 'unsigned int')
        offset_in = Variable('offset_in', 'unsigned int', value=0)
        offset_out = Variable('offset_out', 'unsigned int', value=0)
        stride_in = Variable('stride_in', 'const size_t', array=True)
        stride_out = Variable('stride_out', 'const size_t', array=True)

        kvars, kwvars = common_variables(length, params, 0)
        kwvars.update({
            'tile': tile,
            'offset_in': offset_in,
            'offset_out': offset_out,
            'stride_in': stride_in,
            'stride_out': stride_out})

        stmts = StatementList()
        stmts += Declarations(tile, self.lds_row_padding)

        stmts += If(self.sbrc_type == 'SBRC_3D_FFT_ERC_TRANS_Z_XY',
                    StatementList(
                        Assign(self.lds_row_padding, 1)))

        stmts += Assign(tile, kvars.block_id)
        stmts += Assign(kvars.thread, kvars.thread_id)
        stmts += If(self.sbrc_type == 'SBRC_2D',
                    self.calculate_offsets_2d(**kwvars))
        stmts += If(self.sbrc_type == 'SBRC_3D_FFT_TRANS_XY_Z',
                    self.calculate_offsets_fft_trans_xy_z(**kwvars))
        stmts += If(Or(self.sbrc_type == 'SBRC_3D_FFT_TRANS_Z_XY',
                       self.sbrc_type == 'SBRC_3D_FFT_ERC_TRANS_Z_XY'),
                    self.calculate_offsets_fft_trans_z_xy(**kwvars))

        stmts += Assign(kvars.batch, 0)  # XXX

        return stmts


    def load_from_global(self, buf=None, offset=None, lds=None,
                         lengths=None, thread=None, thread_id=None, stride=None, stride0=None,
                         **kwargs):

        length, params = self.length, self.params
        kvars, kwvars = common_variables(length, params, 0)
        tile = Variable('tile', 'unsigned int')
        offset_in = Variable('offset_in', 'unsigned int', value=0)
        offset_out = Variable('offset_out', 'unsigned int', value=0)
        stride_in = Variable('stride_in', 'const size_t', array=True)
        stride_out = Variable('stride_out', 'const size_t', array=True)

        height = (length * self.rows_per_tile) // params.threads_per_block
#        assert(height < self.rows_per_tile)

        stmts = StatementList()
        stmts += Assign(kvars.thread, kvars.thread_id)

        # SBRC_2D, SBRC_3D_FFT_TRANS_Z_XY, SBRC_3D_FFT_ERC_TRANS_Z_XY
        load = StatementList()
        for h in range(height):
            lidx = kvars.thread
            lidx += h * B(params.threads_per_block + self.n_device_calls * self.lds_row_padding)
            lidx += B(kvars.thread / length) * self.lds_row_padding
            gidx = offset_in + kvars.thread + h * params.threads_per_block
            load += Assign(kvars.lds[lidx], LoadGlobal(kvars.buf, gidx))
        stmts += If(Or(self.sbrc_type == 'SBRC_2D',
                       self.sbrc_type == 'SBRC_3D_FFT_TRANS_Z_XY',
                       self.sbrc_type == 'SBRC_3D_FFT_ERC_TRANS_Z_XY'), load)

        # SBRC_3D_FFT_TRANS_XY_Z
        tiles_per_batch = B(lengths[1] * B(B(lengths[2] + self.rows_per_tile - 1) / self.rows_per_tile))
        tile_in_batch = tile % tiles_per_batch
        load = StatementList()
        for h in range(height):
            lidx = h % height * length
            lidx += h // height * length
            lidx += kvars.thread % length
            lidx += B(kvars.thread / length) * (height * length)
            gidx = offset_in
            gidx += B(kvars.thread % length) * stride_in[0]
            gidx += B(B(kvars.thread / length) * height + h) * stride_in[2]
            idx = tile_in_batch / lengths[1] * self.rows_per_tile + h + thread / length * self.rows_per_tile / params.threads_per_block
            load += If(Or(self.transpose_type != 'TILE_UNALIGNED', idx < lengths[2]),
                       StatementList(Assign(kvars.lds[lidx], LoadGlobal(kvars.buf, gidx))))
        stmts += If(self.sbrc_type == 'SBRC_3D_FFT_TRANS_XY_Z', load)

        return stmts


    def store_to_global(self, stride=None, stride0=None, thread=None,
                        lengths=None, buf=None, offset=None, lds=None,
                        **kwargs):
        length, params = self.length, self.params
        kvars, kwvars = common_variables(length, params, 0)
        tile = Variable('tile', 'unsigned int')
        offset_in = Variable('offset_in', 'unsigned int', value=0)
        offset_out = Variable('offset_out', 'unsigned int', value=0)
        stride_in = Variable('stride_in', 'const size_t', array=True)
        stride_out = Variable('stride_out', 'const size_t', array=True)

        height = length * self.rows_per_tile // params.threads_per_block

        stmts = StatementList()

        # POSTPROCESSING SBRC_3D_FFT_ERC_TRANS_Z_XY
        post = StatementList()
        null = Variable('nullptr', 'void*')

        for h in range(height * self.n_device_calls):
            post += Call('post_process_interleaved_inplace',
                         templates=TemplateList(kvars.scalar_type, 'true', 'CallbackType::NONE'),
                         arguments=ArgumentList(kvars.thread,
                                                length - kvars.thread,
                                                length,
                                                length // self.n_device_calls,
                                                Address(kvars.lds[h * B(length + self.lds_row_padding)]),
                                                0,
                                                Address(kvars.twiddles[length]),
                                                null, null, 0, null, null))
        post += SyncThreads()
        stmts += If(self.sbrc_type == 'SBRC_3D_FFT_ERC_TRANS_Z_XY', post)

        # SBRC_2D
        store = StatementList()
        for h in range(height):
            row = B(B(kvars.thread + h * params.threads_per_block) / self.rows_per_tile)
            col = B(kvars.thread % self.rows_per_tile)
            lidx = col * length + row
            gidx = offset_out + row * stride_out[0] + col * stride_out[1]
            store += StoreGlobal(kvars.buf, gidx, lds[lidx])
        stmts += If(self.sbrc_type == 'SBRC_2D', store)

        # SBRC_3D_FFT_TRANS_XY_Z
        store = StatementList()
        tiles_per_batch = B(lengths[1] * B(B(lengths[2] + self.rows_per_tile - 1) / self.rows_per_tile))
        tile_in_batch = tile % tiles_per_batch

        for h in range(height):
            lidx = h * height
            lidx += B(kvars.thread % self.rows_per_tile) * length
            lidx += B(kvars.thread / self.rows_per_tile)
            gidx = offset_out
            gidx += B(kvars.thread % self.rows_per_tile) * stride_out[0]
            gidx += B(B(kvars.thread / self.rows_per_tile) + h * height) * stride_out[1]
            idx = tile_in_batch / lengths[1] * self.rows_per_tile + thread % self.rows_per_tile
            store += If(Or(self.transpose_type != 'TILE_UNALIGNED', idx < lengths[2]),
                        StatementList(StoreGlobal(kvars.buf, gidx, lds[lidx])))
        stmts += If(self.sbrc_type == 'SBRC_3D_FFT_TRANS_XY_Z', store)

        # SBRC_3D_FFT_TRANS_Z_XY, SBRC_3D_FFT_ERC_TRANS_Z_XY
        store = StatementList()
        for h in range(height):
            lidx = h * height
            lidx += B(kvars.thread % self.rows_per_tile) * B(length + self.lds_row_padding)
            lidx += B(kvars.thread / self.rows_per_tile)
            gidx = offset_out
            gidx += B(kvars.thread % self.rows_per_tile) * stride_out[0]
            gidx += B(B(kvars.thread / self.rows_per_tile) + h * height) * stride_out[2]
            store += StoreGlobal(kvars.buf, gidx, lds[lidx])

        h = height
        lidx = h * height
        lidx += B(kvars.thread % self.rows_per_tile) * B(length + self.lds_row_padding)
        lidx += B(kvars.thread / self.rows_per_tile)
        gidx = offset_out
        gidx += B(kvars.thread % self.rows_per_tile) * stride_out[0]
        gidx += B(B(kvars.thread / self.rows_per_tile) + h * height) * stride_out[2]
        store += If(And(self.sbrc_type == 'SBRC_3D_FFT_ERC_TRANS_Z_XY',
                        kvars.thread < self.rows_per_tile),
                    StatementList(
                        StoreGlobal(kvars.buf, gidx, lds[lidx])))
        stmts += If(Or(self.sbrc_type == 'SBRC_3D_FFT_TRANS_Z_XY',
                       self.sbrc_type == 'SBRC_3D_FFT_ERC_TRANS_Z_XY'),
                    store)

        return stmts





class StockhamTilingCR(StockhamTiling):
    """Column/row tiling."""

    name = 'SBCR'

    def __init__(self, factors, params, **kwargs):
        super().__init__(factors, params, **kwargs)
        self.tile_index = Variable('tile_index', 'size_t')
        self.tile_length = Variable('tile_length', 'size_t')
        self.edge = Variable('edge', 'bool')
        self.tid1 = Variable('tid1', 'size_t')
        self.tid0 = Variable('tid0', 'size_t')

    def calculate_offsets(self, transform=None, dim=None,
                          block_id=None, thread_id=None, lengths=None, stride=None, offset=None, batch=None,
                          offset_lds=None,
                          **kwargs):

        length, params = self.length, self.params

        d             = Variable('d', 'int')
        index_along_d = Variable('index_along_d', 'size_t')
        remaining     = Variable('remaining', 'size_t')
        plength       = Variable('plength', 'size_t', value=1)

        stmts = StatementList()
        stmts += Declarations(self.tile_index, self.tile_length)
        stmts += LineBreak()
        stmts += CommentLines('calculate offset for each tile:',
                              '  tile_index  now means index of the tile along dim1',
                              '  tile_length now means number of tiles along dim1')
        stmts += Declarations(plength, remaining, index_along_d)
        stmts += Assign(self.tile_length, B(lengths[1] - 1) / params.transforms_per_block + 1)
        stmts += Assign(plength, self.tile_length)
        stmts += Assign(self.tile_index, block_id % self.tile_length)
        stmts += Assign(remaining, block_id / self.tile_length)
        stmts += Assign(offset, self.tile_index * params.transforms_per_block * stride[1])
        stmts += For(d.inline(2), d < dim, Increment(d),
                     StatementList(
                         Assign(plength, plength * lengths[d]),
                         Assign(index_along_d, remaining % lengths[d]),
                         Assign(remaining, remaining / lengths[d]),
                         Assign(offset, offset + index_along_d * stride[d])))
        stmts += LineBreak()
        stmts += Assign(transform, self.tile_index * params.transforms_per_block + thread_id / params.threads_per_transform)
        stmts += Assign(batch, block_id / plength)
        stmts += Assign(offset, offset + batch * stride[dim])
        stmts += Assign(offset_lds, length * B(transform % params.transforms_per_block))

        return stmts

    def load_from_global(self, buf=None, offset=None, lds=None,
                         lengths=None, thread_id=None, stride=None, stride0=None, **kwargs):

        length, params = self.length, self.params

        edge, tid0, tid1 = self.edge, self.tid0, self.tid1
        stripmine_w   = params.transforms_per_block
        stripmine_h   = params.threads_per_block // stripmine_w
        stride_lds    = length + kwargs.get('lds_padding', 0)  # XXX

        stmts = StatementList()
        stmts += Declarations(edge, tid0, tid1)
        stmts += ConditionalAssign(edge,
                                   Greater(B(self.tile_index + 1) * params.transforms_per_block, lengths[1]),
                                   'true', 'false')
        stmts += Assign(tid1, thread_id % stripmine_w)  # tid0 walks the columns; tid1 walks the rows
        stmts += Assign(tid0, thread_id / stripmine_w)
        offset_tile_rbuf = lambda i : tid1 * stride[1]  + B(tid0 + i * stripmine_h) * stride0
        offset_tile_wlds = lambda i : tid1 * stride_lds + B(tid0 + i * stripmine_h) * 1
        pred, tmp_stmts = StatementList(), StatementList()
        pred = self.tile_index * params.transforms_per_block + tid1 < lengths[1]
        for i in range(length // stripmine_h):
            tmp_stmts += Assign(lds[offset_tile_wlds(i)], LoadGlobal(buf, offset + offset_tile_rbuf(i)))

        stmts += If(Not(edge), tmp_stmts)
        stmts += If(edge, If(pred, tmp_stmts))

        return stmts

    def store_to_global(self, stride=None, stride0=None, lengths=None, thread_id=None, buf=None,
                        offset=None, lds=None, lds_padding=None, **kwargs):

        length, params = self.length, self.params
        kvars, kwvars = common_variables(length, params, 0)
        edge, tid0, tid1 = self.edge, self.tid0, self.tid1
        stripmine_w   = params.transforms_per_block
        stripmine_h   = params.threads_per_block // stripmine_w
        stride_lds    = length + kwargs.get('lds_padding', 0)  # XXX

        lds_strip_h   = params.threads_per_block // length

        stmts = StatementList()
        offset_tile_rbuf = lambda i : i * lds_strip_h * stride[1] + tid1 * stride[1] + tid0 * stride0
        offset_tile_wlds = lambda i : i * params.threads_per_block + tid1 * B(length + lds_padding) + tid0
        offset_tile_wbuf = offset_tile_rbuf
        offset_tile_rlds = offset_tile_wlds

        stmts += Assign(tid0, thread_id % length)
        stmts += Assign(tid1, thread_id / length)

        pred, tmp_stmts = StatementList(), StatementList()
        pred = self.tile_index * params.transforms_per_block + tid1 < lengths[1]
        for i in range(params.transforms_per_block // lds_strip_h):
            tmp_stmts += StoreGlobal(buf, offset + offset_tile_wbuf(i), lds[offset_tile_rlds(i)])
        stmts += If(Not(edge), tmp_stmts)
        stmts += If(edge, If(pred, tmp_stmts))

        return stmts



#
# Kernel helpers
#

def add_work(generator, guard=False, write=None,
             length=None, width=None, height=None,
             thread=None, threads_per_transform=None,
             **kwargs):
    """Call `generator` as many times as needed..."""
    stmts = StatementList()
    iheight = floor(height)

    if height > iheight and threads_per_transform > length // width:
        iheight += 1
    # else:
    #     guard = False

    kwargs = kwargs.copy()
    kwargs.update(dict(length=length, width=width, height=height, iheight=iheight,
                       thread=thread, threads_per_transform=threads_per_transform))

    work = StatementList()
    for h in range(iheight):
        work += generator(h, **kwargs)

    if guard:
        stmts += CommentLines('more than enough threads, some do nothing')
        if threads_per_transform != length // width:
            stmts += If(And(write, thread < length // width), work)
        else:
            stmts += If(write, work)
    else:
        stmts += work

    if height > iheight and threads_per_transform < length // width:
        stmts += CommentLines('not enough threads, some threads do extra work')
        stmts += If(And(write, thread + iheight * threads_per_transform < length // width),
                    generator(0, hr=iheight, dt=iheight*threads_per_transform, **kwargs))
    stmts += LineBreak()
    return stmts


def load_lds_generator(h, hr=None, dt=0, threads_per_transform=None,
                       thread=None, length=None, width=None, component=None,
                       R=None, lds=None, offset_lds=None, lstride=None, **kwargs):
    """Load registers 'R' from LDS 'X'."""
    hr = hr or h
    load = StatementList()
    for w in range(width):
        tid = B(thread + dt + h * threads_per_transform)
        idx = offset_lds + B(tid + w * (length // width)) * lstride
        if component is not None:
            load += Assign(getattr(R[hr * width + w], component), lds[idx])
        else:
            load += Assign(R[hr * width + w], lds[idx])
    return load


def load_global_generator(h, hr=None, dt=0, threads_per_transform=None,
                          thread=None, length=None, width=None,
                          R=None, buf=None, offset=None, stride0=None, **kwargs):
    """Load registers 'R' from global 'buf'."""
    hr = hr or h
    load = StatementList()
    for w in range(width):
        tid = B(thread + dt + h * threads_per_transform)
        idx = B(tid + w * (length // width))
        load += Assign(R[hr * width + w], LoadGlobal(buf, offset + B(idx) * stride0))
    return load


def apply_twiddle_generator(h, hr=None, dt=0, threads_per_transform=None,
                            thread=None, width=None, cumheight=None,
                            W=None, t=None, twiddles=None, R=None, **kwargs):
    """Apply twiddles from 'T' to registers 'R'."""

    hr = hr or h
    work = StatementList()
    for w in range(1, width):
        tid  = B(thread + dt + h * threads_per_transform)
        tidx = cumheight - 1 + w - 1 + (width - 1) * B(tid % cumheight)
        ridx = hr * width + w
        work += Assign(W, twiddles[tidx])
        work += Assign(t, TwiddleMultiply(R[ridx], W))
        work += Assign(R[ridx], t)
    return work


def butterfly_generator(h, hr=None, width=None, R=None, **kwargs):
    """Apply butterly to registers 'R'."""
    hr = hr or h
    return Call(name=f'FwdRad{width}B1',
                arguments=ArgumentList(*[R[hr * width + w].address() for w in range(width)]))


def store_lds_generator(h, hr=None, dt=0, threads_per_transform=None,
                        thread=None, width=None, cumheight=None, component=None,
                        lds=None, R=None, offset_lds=None, lstride=None, **kwargs):
    """Store registers 'R' to LDS 'X'."""
    hr = hr or h
    work = StatementList()
    for w in range(width):
        tid  = B(thread + dt + h * threads_per_transform)
        idx = offset_lds + B(B(tid / cumheight) * (width * cumheight) + tid % cumheight + w * cumheight) * lstride
        if component is not None:
            work += Assign(lds[idx], getattr(R[hr * width + w], component))
        else:
            work += Assign(lds[idx], R[hr * width + w])
    return work

def store_global_generator(h, hr=None, dt=0, threads_per_transform=None,
                           thread=None, width=None, cumheight=None,
                           buf=None, R=None, offset=None, stride0=None, **kwargs):
    """Store registers 'R' to LDS 'X'."""
    hr = hr or h
    work = StatementList()
    for w in range(width):
        tid  = B(thread + dt + h * threads_per_transform)
        idx = offset + B(B(tid / cumheight) * (width * cumheight) + tid % cumheight + w * cumheight) * stride0
        work += Assign(buf[idx], R[hr * width + w])
    return work

#
# Stockham kernels
#

class StockhamKernel:
    """Base Stockham kernel."""

    def __init__(self, factors, params, scheme, tiling, large_twiddles, **kwargs):
        self.factors = factors
        self.length = product(factors)
        self.params = params
        self.scheme = scheme
        self.tiling = tiling
        self.large_twiddles = large_twiddles
        self.kwargs = kwargs
        self.n_device_calls = 1
        self.lds_row_padding = 1

    def device_templates(self, kvars, **kwvars):
        templates = TemplateList(kvars.scalar_type, kvars.stride_type)
        templates = self.large_twiddles.add_templates(templates, **kwvars)
        templates = self.tiling.add_templates(templates, **kwvars)
        return templates

    def device_call_templates(self, kvars, **kwvars):
        templates = TemplateList(kvars.scalar_type, kvars.stride_type)
        templates = self.large_twiddles.add_templates(templates, **kwvars)
        templates = self.tiling.add_templates(templates, **kwvars)
        return templates

    def global_templates(self, kvars, **kwvars):
        templates = TemplateList(kvars.scalar_type, kvars.stride_type, kvars.embedded_type, kvars.callback_type)
        templates = self.large_twiddles.add_templates(templates, **kwvars)
        templates = self.tiling.add_templates(templates, **kwvars)
        return templates

    def device_arguments(self, kvars, **kwvars):
        arguments = ArgumentList(kvars.lds, kvars.twiddles, kvars.stride_lds, kvars.offset_lds, kvars.write)
        arguments = self.large_twiddles.add_device_arguments(arguments, **kwvars)
        arguments = self.tiling.add_device_arguments(arguments, **kwvars)
        return arguments

    def device_call_arguments(self, kvars, **kwvars):
        arguments = ArgumentList(kvars.lds, kvars.twiddles, kvars.stride_lds, kvars.offset_lds, kvars.write)
        arguments = self.large_twiddles.add_device_call_arguments(arguments, **kwvars)
        arguments = self.tiling.add_device_call_arguments(arguments, **kwvars)
        return arguments

    def global_arguments(self, kvars, **kwvars):
        cb_args = get_callback_args()
        arguments = ArgumentList(kvars.twiddles, kvars.dim, kvars.lengths, kvars.stride, kvars.nbatch, kvars.lds_padding) \
                  + cb_args + ArgumentList(kvars.buf)
        arguments = self.large_twiddles.add_global_arguments(arguments, **kwvars)
        arguments = self.tiling.add_global_arguments(arguments, **kwvars)
        return arguments

    def generate_device_function(self):
        """Stockham device function."""
        pass

    def generate_global_function(self, **kwargs):
        """Global Stockham function."""

        use_3steps = kwargs.get('3steps')
        length, params = self.length, self.params

        kvars, kwvars = common_variables(self.n_device_calls * self.length, params, self.nregisters)

        body = StatementList()
        body += CommentLines(
            f'this kernel:',
            f'  uses {params.threads_per_transform} threads per transform',
            f'  does {params.transforms_per_block} transforms per thread block',
            f'therefore it should be called with {params.threads_per_block} threads per thread block')
        body += Declarations(kvars.lds_uchar, kvars.lds,
                             kvars.offset, kvars.offset_lds, kvars.stride_lds,
                             kvars.batch, kvars.transform, kvars.thread, kvars.write)
        body += Declaration(kvars.stride0.name, kvars.stride0.type,
                            value=Ternary(kvars.stride_type == 'SB_UNIT', 1, kvars.stride[0]))
        body += CallbackDeclaration()

        body += LineBreak()
        body += CommentLines('large twiddles')
        body += self.large_twiddles.load(self.length, params, **kwvars)

        body += LineBreak()
        body += CommentLines('offsets')
        body += self.tiling.calculate_offsets(**kwvars)

        body += LineBreak()
        body += If(GreaterEqual(kvars.batch, kvars.nbatch), [ReturnStatement()])

        body += LineBreak()
        body += CommentLines('load global')
        body += self.tiling.load_from_global(**kwvars)

        body += LineBreak()
        body += CommentLines('handle even-length real to complex pre-process in lds before transform')
        body += self.tiling.real2cmplx_pre_post(self.length, True, params, **kwvars)

        body += LineBreak()
        body += CommentLines('transform')
        body += Assign(kvars.write, 'true')
        for c in range(self.n_device_calls):
            templates = self.device_call_templates(kvars, **kwvars)
            arguments = self.device_call_arguments(kvars, **kwvars)

            templates.set_value(kvars.stride_type.name, 'SB_UNIT')
            if c > 0:
                arguments.set_value(kvars.offset_lds.name,
                                    kvars.offset_lds + c * B(length + self.lds_row_padding) * params.transforms_per_block)
            body += Call(f'forward_length{self.length}_{self.tiling.name}_device',
                         arguments=arguments, templates=templates)

        body += LineBreak()
        body += CommentLines('handle even-length complex to real post-process in lds after transform')
        body += self.tiling.real2cmplx_pre_post(self.length, False, params, **kwvars)

        body += LineBreak()
        body += CommentLines('store global')
        body += SyncThreads()
        body += self.tiling.store_to_global(**kwvars)

        return Function(name=f'forward_length{self.length}_{self.tiling.name}',
                        qualifier='__global__',
                        launch_bounds=params.threads_per_block,
                        arguments=self.global_arguments(kvars, **kwvars),
                        templates=self.global_templates(kvars, **kwvars),
                        meta=NS(factors=self.factors,
                                length=self.length,
                                transforms_per_block=params.transforms_per_block,
                                threads_per_block=params.threads_per_block,
                                params=params,
                                scheme=self.scheme,
                                use_3steps_large_twd=use_3steps,
                                transpose=None),
                        body=body)


class StockhamKernelConsolidated(StockhamKernel):
    """Consolidated device..."""

    @property
    def nregisters(self):
        registers = [ceil(self.length / width / self.params.threads_per_transform) * width for width in self.factors]
        return max(registers)

    def generate_device_function(self, **kwargs):
        factors, length, params = self.factors, self.length, self.params
        kvars, kwvars = common_variables(length, params, self.nregisters)

        body = StatementList()
        body += Declarations(kvars.thread, kvars.R, kvars.W, kvars.t)
        body += Declaration(kvars.lstride.name, kvars.lstride.type,
                            value=Ternary(kvars.stride_type == 'SB_UNIT', 1, kvars.stride_lds))

        if length == 1:
            return Function(f'forward_length{length}_{self.tiling.name}_device',
                            arguments=self.device_arguments(kvars, **kwvars),
                            templates=self.device_templates(kvars, **kwvars),
                            body=StatementList(),
                            qualifier='__device__',
                            meta=NS(factors=self.factors,
                                    length=self.length,
                                    params=params,
                                    nregisters=self.nregisters,
                                    flavour='consolidated'))


        body = StatementList()
        body += Declarations(kvars.thread, kvars.R, kvars.W, kvars.t)
        body += Declaration(kvars.lstride.name, kvars.lstride.type,
                            value=Ternary(kvars.stride_type == 'SB_UNIT', 1, kvars.stride_lds))

        body += Assign(kvars.thread, kvars.thread_id % params.threads_per_transform)

        for npass, width in enumerate(factors):
            height = length // width / params.threads_per_transform
            cumheight = product(factors[:npass])

            kwvars.update(length=length, width=width, height=height, cumheight=cumheight,
                          threads_per_transform=params.threads_per_transform)

            body += LineBreak()
            body += CommentLines(f'pass {npass}, width {width}',
                                 f'using {params.threads_per_transform} threads we need to do {length // width} radix-{width} butterflies',
                                 f'therefore each thread will do {height} butterflies')
            body += SyncThreads()
            body += add_work(load_lds_generator, **kwvars)

            if npass > 0:
                body += add_work(apply_twiddle_generator, **kwvars)

            body += add_work(butterfly_generator, **kwvars)

            if npass == len(factors) - 1:
                body += self.large_twiddles.multiply(**kwvars)

            body += SyncThreads()
            body += add_work(store_lds_generator, guard=True, **kwvars)

        return Function(f'forward_length{length}_{self.tiling.name}_device',
                        arguments=self.device_arguments(kvars, **kwvars),
                        templates=self.device_templates(kvars, **kwvars),
                        body=body,
                        qualifier='__device__',
                        meta=NS(factors=self.factors,
                                length=self.length,
                                params=params,
                                nregisters=self.nregisters,
                                flavour='consolidated'))


class StockhamKernelFused2D(StockhamKernel):

    def __init__(self, device_functions):
        self.tiling = StockhamTiling([], None)
        self.large_twiddles = StockhamLargeTwiddles()
        self.device_functions = device_functions

        kernels = self.device_functions
        length = [ x.meta.length for x in self.device_functions ]
        tpt = max(length[1]*kernels[0].meta.params.threads_per_transform,
                  length[0]*kernels[1].meta.params.threads_per_transform)
        tpb = max(kernels[0].meta.params.threads_per_block,
                  kernels[1].meta.params.threads_per_block)
        self.params =  get_launch_params(length,
                                         threads_per_transform=tpt,
                                         threads_per_block=tpb)


    def generate_global_function(self, **kwargs):

        params, kernels = self.params, self.device_functions
        length = [ x.meta.length for x in kernels ]
        kvars, kwvars = common_variables(product(length), params,
                                         max(x.meta.nregisters for x in kernels))

        body = StatementList()
        body += LineBreak()
        body += CommentLines(
            '',
            f'this kernel:',
            f'  uses {params.threads_per_transform} threads per 2d transform',
            f'  does {params.transforms_per_block} 2d transforms per thread block',
            f'therefore it should be called with {params.threads_per_block} threads per thread block',
            '')

        d             = Variable('d', 'int')
        index_along_d = Variable('index_along_d', 'size_t')
        remaining     = Variable('remaining', 'size_t')
        plength       = Variable('plength', 'size_t', value=1)

        stride0 = Variable('stride0', 'size_t')
        batch0  = Variable('batch0', 'size_t')
        batch1  = Variable('batch1', 'size_t')
        write   = Variable('write', 'bool')

        body += Declarations(kvars.lds_uchar, kvars.lds,
                             kvars.thread, kvars.transform,
                             kvars.offset, kvars.offset_lds, kvars.stride_lds, kvars.write,
                             stride0, batch0, batch1, remaining, plength, d, index_along_d)
        body += CallbackDeclaration()

        # load
        body += LineBreak()
        body += CommentLines('', f'length: {length[0]}', '')

        body += LineBreak()
        tpb = length[1] * params.transforms_per_block
        body += CommentLines(f'transform is: length {length[0]} transform number',
                             f'there are {length[1]} * {params.transforms_per_block} = {tpb} of them per block')
        body += Assign(kvars.transform, kvars.block_id * tpb + kvars.thread_id / (params.threads_per_block // tpb))
        body += Assign(remaining, kvars.transform)
        body += For(InlineAssign(d, 1), d < kvars.dim, Increment(d),
                    StatementList(
                        Assign(plength, plength * kvars.lengths[d]),
                        Assign(index_along_d, remaining % kvars.lengths[d]),
                        Assign(remaining, remaining / kvars.lengths[d]),
                        Assign(kvars.offset, kvars.offset + index_along_d * kvars.stride[d])))
        body += Assign(batch0, kvars.transform / plength)
        body += Assign(kvars.offset, kvars.offset + batch0 * kvars.stride[kvars.dim])

        body += LineBreak()
        body += CommentLines(f'load following length {length[0]}')
        width = length[0] // kernels[0].meta.params.threads_per_transform
        height = kernels[0].meta.params.threads_per_transform
        body += Assign(kvars.write, batch0 < kvars.nbatch)
        body += Assign(kvars.thread, kvars.thread_id % height)
        body += Assign(stride0, kvars.stride[0])
        body += Assign(kvars.offset_lds, length[0] * B(kvars.transform % (length[1] * params.transforms_per_block)))
        stmts = StatementList()
        for w in range(width):
            idx = kvars.thread + w * height
            stmts += Assign(kvars.lds[kvars.offset_lds + idx], LoadGlobal(kvars.buf, kvars.offset + B(idx) * stride0))
        body += If(kvars.write, stmts)

        templates = self.device_call_templates(kvars, **kwvars)
        templates.set_value(kvars.stride_type.name, 'SB_UNIT')
        body += Assign(kvars.stride_lds, 1)
        body += kernels[0].call(self.device_call_arguments(kvars, **kwvars), templates)

        # note there is a syncthreads at the start of the next call

        body += LineBreak()
        body += CommentLines('', f'length: {length[1]}', '')
        body += LineBreak()

        width = length[1] // kernels[1].meta.params.threads_per_transform
        height = kernels[1].meta.params.threads_per_transform
        tpb = length[0] * params.transforms_per_block
        body += CommentLines(f'transform is: length {length[1]} transform number',
                             f'there are {length[0]} * {params.transforms_per_block} = {tpb} of them per block')

        body += Assign(kvars.transform, kvars.block_id * tpb + kvars.thread_id / (params.threads_per_block // tpb))
        body += Assign(plength, kvars.lengths[0])
        body += For(InlineAssign(d, 2), d < kvars.dim, Increment(d),
                    StatementList(
                        Assign(plength, plength * kvars.lengths[d])))
        body += Assign(batch1, kvars.transform / plength)
        body += Assign(kvars.write, batch1 < kvars.nbatch)
        body += Assign(kvars.thread, kvars.thread_id % height)
        body += Assign(kvars.offset_lds, (length[0]*length[1]) * B(B(kvars.transform % tpb) / length[0]) + kvars.transform % length[0])

        templates = self.device_call_templates(kvars, **kwvars)
        templates.set_value(kvars.stride_type.name, 'SB_NONUNIT')
        arguments = self.device_call_arguments(kvars, **kwvars)
        if kernels[0].meta.factors != kernels[1].meta.factors:
            arguments.set_value(kvars.twiddles.name, kvars.twiddles + length[0])
        body += Assign(kvars.stride_lds, length[0])
        body += kernels[1].call(arguments, templates)

        # store
        body += LineBreak()
        body += CommentLines(f'store following length {length[0]}')
        body += SyncThreads()
        tpb = length[1] * params.transforms_per_block
        body += Assign(kvars.transform, kvars.block_id * tpb + kvars.thread_id / (params.threads_per_block // tpb))

        width = length[0] // kernels[0].meta.params.threads_per_transform
        height = kernels[0].meta.params.threads_per_transform

        body += Assign(kvars.write, batch0 < kvars.nbatch)
        body += Assign(kvars.thread, kvars.thread_id % height)
        body += Assign(kvars.offset_lds, length[0] * B(kvars.transform % (length[1] * params.transforms_per_block)))
        stmts = StatementList()
        for w in range(width):
            idx = kvars.thread + w * height
            stmts += StoreGlobal(kvars.buf, kvars.offset + B(idx) * stride0, kvars.lds[kvars.offset_lds + idx])
        body += If(kvars.write, stmts)

        template_list = TemplateList(kvars.scalar_type, kvars.stride_type)
        argument_list = ArgumentList(kvars.twiddles, kvars.dim, kvars.lengths, kvars.stride, kvars.nbatch, kvars.buf)
        return Function(name=f'forward_length{"x".join(map(str, length))}',
                        qualifier='__global__',
                        launch_bounds=params.threads_per_block,
                        templates=self.global_templates(kvars),
                        arguments=self.global_arguments(kvars),
                        meta=NS(length=tuple(length),
                                factors=kernels[0].meta.factors + kernels[1].meta.factors,
                                transforms_per_block=params.transforms_per_block,
                                threads_per_block=params.threads_per_block,
                                params=params,
                                scheme='CS_KERNEL_2D_SINGLE',
                                transpose='NONE'),
                        body=body)



#
# AST transforms
#

def make_variants(kdevice, kglobal):
    """Given in-place complex-interleaved kernels, create all other variations.

    The ASTs in 'kglobal' and 'kdevice' are assumed to be in-place,
    complex-interleaved kernels.

    Return out-of-place and planar variations.
    """
    op_names = ['buf', 'stride', 'stride0', 'offset']

    def rename(x, pre):
        if 'forward' in x or 'inverse' in x:
            return pre + x
        return x

    def rename_ip(x):
        return rename_functions(x, lambda n: rename(n, 'ip_'))

    def rename_op(x):
        return rename_functions(x, lambda n: rename(n, 'op_'))

    if kglobal.meta.scheme == 'CS_KERNEL_STOCKHAM_BLOCK_RC' or kglobal.meta.scheme == 'CS_KERNEL_STOCKHAM_BLOCK_CR':
        # out-of-place only
        kernels = [
            rename_op(make_out_of_place(kdevice, op_names)),
            rename_op(make_out_of_place(kglobal, op_names)),
            rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_out')),
            rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_in')),
            rename_op(make_planar(make_planar(make_out_of_place(kglobal, op_names), 'buf_out'), 'buf_in'))
            ]
        kdevice = make_inverse(kdevice)
        kglobal = make_inverse(kglobal)
        kernels += [
            rename_op(make_out_of_place(kdevice, op_names)),
            rename_op(make_out_of_place(kglobal, op_names)),
            rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_out')),
            rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_in')),
            rename_op(make_planar(make_planar(make_out_of_place(kglobal, op_names), 'buf_out'), 'buf_in')),
        ]
        return kernels

    # XXX: Don't need in-place/out-of-place device functions anymore
    kernels = [
        # in-place, interleaved
        rename_ip(kdevice),
        rename_ip(kglobal),
        # in-place, planar
        rename_ip(make_planar(kglobal, 'buf')),
        # out-of-place, interleaved -> interleaved
        rename_op(make_out_of_place(kdevice, op_names)),
        rename_op(make_out_of_place(kglobal, op_names)),
        # out-of-place, interleaved -> planar
        rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_out')),
        # out-of-place, planar -> interleaved
        rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_in')),
        # out-of-place, planar -> planar
        rename_op(make_planar(make_planar(make_out_of_place(kglobal, op_names), 'buf_out'), 'buf_in')),
    ]

    kdevice = make_inverse(kdevice)
    kglobal = make_inverse(kglobal)

    kernels += [
        # in-place, interleaved
        rename_ip(kdevice),
        rename_ip(kglobal),
        # in-place, planar
        rename_ip(make_planar(kglobal, 'buf')),
        # out-of-place, interleaved -> interleaved
        rename_op(make_out_of_place(kdevice, op_names)),
        rename_op(make_out_of_place(kglobal, op_names)),
        # out-of-place, interleaved -> planar
        rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_out')),
        # out-of-place, planar -> interleaved
        rename_op(make_planar(make_out_of_place(kglobal, op_names), 'buf_in')),
        # out-of-place, planar -> planar
        rename_op(make_planar(make_planar(make_out_of_place(kglobal, op_names), 'buf_out'), 'buf_in')),
    ]

    return kernels


#
# Interface!
#

def stockham_launch(factors, **kwargs):
    """Launch helper.  Not used by rocFFT proper."""

    length = product(factors)
    params = get_launch_params(factors, **kwargs)

    # arguments
    scalar_type   = Variable('scalar_type', 'typename')
    callback_type = Variable('CallbackType::NONE', 'CallbackType')
    stride_type   = Variable('SB_UNIT', 'StrideBin')
    inout         = Variable('inout', 'scalar_type', array=True)
    twiddles      = Variable('twiddles', 'const scalar_type', array=True)
    stride_in     = Variable('stride_in', 'size_t')
    stride_out    = Variable('stride_out', 'size_t')
    nbatch        = Variable('nbatch', 'size_t')
    lds_padding   = Variable('lds_padding', 'unsigned int')
    kargs         = Variable('kargs', 'size_t*')

    # locals
    nblocks = Variable('nblocks', 'int')
    null    = Variable('nullptr', 'void*')

    body = StatementList()
    body += Declarations(nblocks)
    body += Assign(nblocks, B(nbatch + (params.transforms_per_block - 1)) / params.transforms_per_block)
    body += Call(f'forward_length{length}_SBRR',
                 arguments = ArgumentList(twiddles, 1, kargs, kargs + 1, nbatch, lds_padding, null, null, 0, null, null, inout),
                 templates = TemplateList(scalar_type, stride_type, callback_type),
                 launch_params = ArgumentList(nblocks, params.threads_per_block))

    return Function(name = f'forward_length{length}_launch',
                    templates = TemplateList(scalar_type),
                    arguments = ArgumentList(inout, nbatch, lds_padding, twiddles, kargs, stride_in, stride_out),
                    body = body)


def stockham_launch2d(length, params, **kwargs):
    """Launch helper.  Not used by rocFFT proper."""

    # arguments
    scalar_type = Variable('scalar_type', 'typename')
    sb          = Variable('SB_NONUNIT', 'StrideBin')
    cbtype      = Variable('CallbackType::NONE', 'CallbackType')
    inout       = Variable('inout', 'scalar_type', array=True)
    twiddles    = Variable('twiddles', 'const scalar_type', array=True)
    stride_in   = Variable('stride_in', 'size_t')
    stride_out  = Variable('stride_out', 'size_t')
    nbatch      = Variable('nbatch', 'size_t')
    lds_padding = Variable('lds_padding', 'unsigned int')
    kargs       = Variable('kargs', 'size_t*')

    # locals
    nblocks = Variable('nblocks', 'int')
    null    = Variable('nullptr', 'void*')

    body = StatementList()
    body += Declarations(nblocks)
    body += Assign(nblocks, B(nbatch + (params.transforms_per_block - 1)) / params.transforms_per_block)
    length = 'x'.join(map(str, length))
    body += Call(f'forward_length{length}',
                 arguments = ArgumentList(twiddles, 2, kargs, kargs+2, nbatch, null, null, 0, null, null, inout),
                 templates = TemplateList(scalar_type, sb, cbtype),
                 launch_params = ArgumentList(nblocks, params.threads_per_block))

    return Function(name = f'forward_length{length}_launch',
                    templates = TemplateList(scalar_type),
                    arguments = ArgumentList(inout, nbatch, lds_padding, twiddles, kargs, stride_in, stride_out),
                    body = body)


def stockham_default_factors(length):
    supported_radixes = [2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 16, 17]
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


def stockham1d(length, **kwargs):
    """Generate Stockham kernels!

    Returns a list of (device, global) function pairs.  This routine
    is essentially a factory...
    """

    factors = kwargs.get('factors', stockham_default_factors(length))
    if 'factors' in kwargs:
        kwargs.pop('factors')

    # assert that factors multiply out to the length
    if functools.reduce(lambda x, y: x * y, factors) != length:
        raise RuntimeError("invalid factors {} for length {}".format(factors, length))

    defualt_3steps = {
        'sp': 'false',
        'dp': 'false'
    }
    kwargs['3steps'] = kwargs.get('use_3steps_large_twd', defualt_3steps)

    scheme = kwargs.get('scheme', 'CS_KERNEL_STOCKHAM')
    if 'scheme' in kwargs:
        kwargs.pop('scheme')

    threads_per_transform = kwargs.get('threads_per_transform', {
        'uwide': length // min(factors),
        'wide': length // max(factors),
        'tall': None,
        'consolidated': None
        }[kwargs.get('flavour', 'uwide')])
    kwargs['threads_per_transform'] = threads_per_transform

    params = get_launch_params(factors, **kwargs)

    tiling = {
        'CS_KERNEL_STOCKHAM':          StockhamTilingRR(factors, params, **kwargs),
        'CS_KERNEL_STOCKHAM_BLOCK_CC': StockhamTilingCC(factors, params, **kwargs),
        'CS_KERNEL_STOCKHAM_BLOCK_RC': StockhamTilingRC(factors, params, **kwargs),
        'CS_KERNEL_STOCKHAM_BLOCK_CR': StockhamTilingCR(factors, params, **kwargs),
    }[scheme]

    twiddles = StockhamLargeTwiddles()
    if scheme == 'CS_KERNEL_STOCKHAM_BLOCK_CC':
        twiddles = StockhamLargeTwiddles2Step()

    kernel = StockhamKernelConsolidated(factors, params, scheme, tiling, twiddles, **kwargs)

    tiling.update_kernel_settings(kernel)

    kdevice = kernel.generate_device_function(**kwargs)
    kglobal = kernel.generate_global_function(**kwargs)

    return kdevice, kglobal


def stockham2d(lengths, **kwargs):
    """Generate fused 2D Stockham kernel."""

    factorss = kwargs.get('factors', [stockham_default_factors(l) for l in lengths])
    if 'factors' in kwargs:
        kwargs.pop('factors')

    flavours = kwargs.get('flavour', ['uwide' for l in lengths])
    if 'flavour' in kwargs:
        kwargs.pop('flavour')
    if 'scheme' in kwargs:
        kwargs.pop('scheme')

    device_functions = []
    for length, factors, flavour in zip(lengths, factorss, flavours):
        kdevice, _ = stockham1d(length, factors=factors, flavour=flavour, scheme='CS_KERNEL_STOCKHAM', **kwargs)
        device_functions.append(kdevice)

    kernel = StockhamKernelFused2D(device_functions)
    kglobal = kernel.generate_global_function(**kwargs)

    if device_functions[0].name == device_functions[1].name:
        device_functions.pop()
    return StatementList(*device_functions), kglobal


def stockham(length, **kwargs):
    """Generate Stockham kernels!"""

    if isinstance(length, int):
        return stockham1d(length, **kwargs)

    if isinstance(length, (tuple, list)):
        return stockham2d(length, **kwargs)

    raise ValueError("length must be an interger or list")

def stockham_rtc(kernel_prelude, specs, **kwargs):
    """Generate runtime-compile-able stockham kernel source.

    Accepts a namespace object of kernel parameters, returns
    unformatted stringified device+global functions.

    Key differences of RTC kernels:
    - global function is de-templatized
    - it's given "C" linkage, so runtime compiler does not need to do
      C++ name mangling
    """
    op_names = ['buf', 'stride', 'stride0', 'offset']

    def rename(x, pre):
        if 'forward' in x or 'inverse' in x:
            return pre + x
        return x

    def rename_ip(x):
        return rename_functions(x, lambda n: rename(n, 'ip_'))

    def rename_op(x):
        return rename_functions(x, lambda n: rename(n, 'op_'))

    kdevice, kglobal = stockham(**kwargs)

    # rewrite the kernel as required to fit the specs
    if specs['direction'] == 1:
        kdevice = make_inverse(kdevice)
        kglobal = make_inverse(kglobal)
    if specs['inplace']:
        if specs['input_is_planar']:
            kdevice = make_planar(kdevice, 'buf')
            kglobal = make_planar(kglobal, 'buf')
    else:
        kdevice = make_out_of_place(kdevice, op_names)
        kglobal = make_out_of_place(kglobal, op_names)
        if specs['input_is_planar']:
            kdevice = make_planar(kdevice, 'buf_in')
            kglobal = make_planar(kglobal, 'buf_in')
        if not(specs['inplace']) and specs['output_is_planar']:
            kdevice = make_planar(kdevice, 'buf_out')
            kglobal = make_planar(kglobal, 'buf_out')

    # convert global kernel into runtime-compiled version
    kglobal = make_rtc(kglobal, specs)
    return ''.join([kernel_prelude, str(kdevice), str(kglobal)])
